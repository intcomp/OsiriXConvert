import os, json, shutil, argparse, re, itertools
from glob import glob
import os.path as osp
from warnings import warn

import nibabel as nib
import numpy as np
from PIL import Image

from pydicom import dcmread
from vlkit.geometry import polygon2mask
from vlkit.dicom.group import group_study, group_series
from vlkit.dicom import get_pixel_to_patient_transformation_matrix
from vlkit.image import normalize
from vlkit.visualization import overlay_mask

from rt_utils import RTStructBuilder

from dicom_utils import (
    parse_osirix_sr,
    read_dicom_info,
    find_osirix_sr,
    osirix_get_reference_uid,
    get_common_prefix,
    build_SOPInstanceUID_lookup_table
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="rtconvert",
        usage="rtconvert path/to/dicoms/",
        description="""Convert annotations to dicom-rt structure set.
        It will search all suported annotations (currently support OsirixSR)
        and corresponding dicom files, and convert annotations into dicom-rt structure.
        """
    )
    parser.add_argument('data_root')
    parser.add_argument('--save-to', default=None)
    return parser.parse_args()


def process(data_dir):
    print(f"Searching dicom files in {data_dir}, this may take a while.")
    dicoms = glob(f"{data_dir}/**/*.dcm", recursive=True)
    if len(dicoms) == 0:
        raise RuntimeError(f"No dicom file found in {data_dir}.")
    else:
        print(f"Found {len(dicoms)} dicom files, gathering their meta data.")

    dicom_info = read_dicom_info(dicoms)
    studies = group_study(dicom_info)

    # loop over studies
    for study_instance_uid, dicoms in studies.items():
        dicom_paths = [dcm.fullpath for dcm in dicom_info]
        study_prefix = get_common_prefix(dicom_paths)
        SOPInstanceUID_lookup_table = build_SOPInstanceUID_lookup_table(dicom_info)
        series_instance_uid2series = group_series(dicom_info)
        # find out all Osirix SR files
        osirix_sr = find_osirix_sr(dicom_info)

        # eliminate all OsirixSR files without an associated dicom
        associated = [osirix_get_reference_uid(osx) in SOPInstanceUID_lookup_table for osx in osirix_sr]
        if not all(associated):
            ignored = [osp.basename(osx.fullpath) for osx, ass in zip(osirix_sr, associated) if ass is False]
            ignored_str = ", ".join(ignored)
            warn(f"study \"{study_prefix}\" OsirixSR \"{ignored_str}\" ignored due to unable to find associated dicom")
            osirix_sr = list(itertools.compress(osirix_sr, associated))

        # assign Osirix SR files to series
        # Osirix SR annotations might be annotated on different series, e.g. ADC and T2.
        series_instance_uid2osirixsr = dict()
        for osx in osirix_sr:
            series_instance_uid = SOPInstanceUID_lookup_table[osirix_get_reference_uid(osx)].SeriesInstanceUID
            if series_instance_uid in series_instance_uid2osirixsr:
                series_instance_uid2osirixsr[series_instance_uid].append(osx)
            else:
                series_instance_uid2osirixsr[series_instance_uid] = [osx]

        for series_instance_uid, osirix_sr in series_instance_uid2osirixsr.items():
            # a series and corresponding osirix SR files
            series = sorted(series_instance_uid2series[series_instance_uid], key=lambda x:x['fullpath'])
            osirix_sr = sorted(osirix_sr, key=lambda x : SOPInstanceUID_lookup_table[osirix_get_reference_uid(x)].InstanceNumber)
            tmp_dir = osp.join('/tmp/OsirixSR2dicomrt', f'study-{study_instance_uid}/series-{series_instance_uid}')

            os.makedirs(tmp_dir, exist_ok=True)
            for ds in series:
                fullpath = ds.fullpath
                ds = dcmread(fullpath)
                try:
                    ds.pixel_array
                except:
                    warn(f"\"{fullpath}\" cannot access pixel_array")
                fn = osp.basename(fullpath)
                if not hasattr(ds, 'StudyID'):
                    ds.StudyID = study_instance_uid
                ds.save_as(osp.join(tmp_dir, fn))

            # construct RTStruct
            try:
                rtstruct  = RTStructBuilder.create_new(dicom_series_path=tmp_dir)
            except:
                warn(f"Cannot create RTStructure for {tmp_dir}")
                continue

            h, w = dcmread(series[0].fullpath).pixel_array.shape
            d = len(series)

            # extract ROI and names
            named_masks = dict()

            # z coordinates of the top-left corner of the slices
            # this is used to sort the slices
            # if zs has duplicate values, we cannot use it to sort the slices
            # this happens to some generated/computed series
            # e.g. diffusion of different b-values with the same slice location
            zs = np.array([ds.ImagePositionPatient[2] for ds in series])
            if np.unique(zs).size != zs.size:
                raise RuntimeError(f"Duplicated z coordinates in {series[0].fullpath}. This series is not supported.")
            zorder = np.argsort(np.array([ds.ImagePositionPatient[2] for ds in series]))
            for osx in osirix_sr:
                # corresponding dicom
                corres_dicom = SOPInstanceUID_lookup_table[osirix_get_reference_uid(osx)]
                instance_number = int(corres_dicom.InstanceNumber)
                zord = zorder[instance_number - 1]
                rois = parse_osirix_sr(osx.fullpath)

                if len(rois) == 0:
                    warn(f"\"{osx.fullpath}\" has no ROI, ignored.")
                    continue

                # save polygon coords into json
                with open(f"{corres_dicom.fullpath}.roi.json", "w") as f:
                    json.dump(
                        [{"name": name, "polygon": [c.tolist() for c in coords] } for name, coords in rois.items()],
                        f,
                        indent=4
                    )

                for name, coords in rois.items():
                    # convert polygons to binary mask
                    assert all([c.ndim == 2 for c in coords]), f"OsirixSR {osx.fullpath} dcm {corres_dicom.fullpath}: ROI {name} has non-2D polygon."
                    if len(coords) == 1:
                        mask1 = polygon2mask(coords[0], h, w)
                    elif len(coords) > 1:
                        mask1 = np.logical_or(*[polygon2mask(c, h, w) for c in coords])
                    else:
                        raise RuntimeError(f"{osx.fullpath} ROI name {name} has {len(coords)} polygon.")
                    # save polygon coords
                    try:
                        np.save(corres_dicom.fullpath + f".{name}.poly.npy", np.stack(coords, axis=0))
                    except:
                        warn(f"Cannot save {corres_dicom.fullpath} ROI name {name} polygon coords.")
                        continue
                    # save mask and polygon
                    msk_path = corres_dicom.fullpath + f".{name}.mask"
                    img8 = normalize(dcmread(corres_dicom.fullpath).pixel_array, 0, 255).astype(np.uint8)
                    Image.fromarray(overlay_mask(
                        img8, mask1,
                        alpha=0.2,
                        show_boundary=True, boundary_color=(0, 255, 0))
                    ).save(corres_dicom.fullpath + ".png")
                    #
                    np.save(msk_path + ".npy", mask1)
                    Image.fromarray((mask1 * 255).astype(np.uint8)).save(msk_path + ".png")
                    #
                    if name not in named_masks:
                        named_masks[name] = np.zeros((h, w, d), dtype=bool)
                    named_masks[name][:, :, zord] = mask1

            # if is there any valid ROI
            if len(named_masks) > 0:
                for name, mask in named_masks.items():
                    rtstruct.add_roi(mask, name=name)

                # save the RTStruct
                filename = series[0].SeriesDescription.replace(" ", "-").replace('/', '-').replace('\\', '-')
                filename = re.sub(r'-+', '-', filename) + '_rtstruct.dcm'
                save_path = osp.join(study_prefix, "rt-struct", filename)
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                print(f"Saved structure set to \"{save_path}\"")
                rtstruct.save(save_path)

                # transformation matrix
                trans_matrix = get_pixel_to_patient_transformation_matrix(
                    sorted(series, key=lambda x: x.ImagePositionPatient[2])
                )

                # save mask to Nifti
                # be aware that the Nifti is alwasy in inferior-superior direction
                for name, mask in named_masks.items():
                    nifti_img = nib.Nifti1Image(mask.astype(np.uint8), trans_matrix)
                    nifti_fn = osp.join(study_prefix, f"nifti-{name}.nii.gz")
                    nib.save(nifti_img, nifti_fn)
                    print(f"Saved Nifti for ROI \"{name}\" to \"{nifti_fn}\".")

                # cleanup the temporary directory
                shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    args = parse_args()
    if not osp.isdir(args.data_root):
        raise RuntimeError(f'{args.data_root} is not a directory')
    if args.data_root == '/':
        warn("You are searching dicoms in the root directory, this might be EXTREMELY time-consuming. Consider providing a more specific sub-directory.")
    process(args.data_root)
