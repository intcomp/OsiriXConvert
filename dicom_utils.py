import pydicom
import numpy as np
import os.path as osp
import pathlib
from glob import glob
import logging
from warnings import warn
from tqdm import tqdm
import NSKeyedUnArchiver
from vlkit import Dotdict


def get_logger(log_file):
    logger = logging.getLogger("RTConvert")
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def is_osirix_sr(ds):
    return hasattr(pydicom.dcmread(ds.fullpath), "EncapsulatedDocument")


def osirix_get_reference_uid(ds):
    try:
        ref = pydicom.dcmread(ds.fullpath).ContentSequence[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID
    except Exception as e:
        ref = None
        print(f"Cannot read referred dicom: {e}")
    return ref


def get_common_prefix(paths):
    shortest = 0
    # find the shallowest file path
    for idx, f in enumerate(paths):
        if len(paths[shortest].split(osp.sep)) <len(f.split(osp.sep)):
            shortest = idx
    shortest = paths[shortest].split(osp.sep)
    for i in range(len(shortest), 0, -1):
        path = osp.sep.join(shortest[:i])
        if all([pathlib.PurePath(p).is_relative_to(path) for p in paths]):
            return path


def read_dicom_info(input):
    if isinstance(input, str):
        dicoms = sorted(glob(f"{input}/**/*.dcm", recursive = True))
    else:
        assert isinstance(input, list)
        dicoms = input
    results = []
    for d in tqdm(dicoms):
        try:
            ds = pydicom.dcmread(d)
        except:
            warn(f"{d} is not a valid dicom file")
            continue
        InstanceNumber = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else None
        ds = dict(
            fullpath=d,
            SeriesDescription=ds.SeriesDescription if hasattr(ds, "SeriesDescription") else "",
            SeriesInstanceUID=ds.SeriesInstanceUID,
            SOPInstanceUID=ds.SOPInstanceUID,
            StudyInstanceUID=ds.StudyInstanceUID,
            InstanceNumber=InstanceNumber,
            PixelSpacing=ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else None,
            SliceLocation=float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None,
            ImageOrientationPatient=np.array(ds.ImageOrientationPatient) if hasattr(ds, 'ImageOrientationPatient') else None,
            ImagePositionPatient=np.array(ds.ImagePositionPatient) if hasattr(ds, 'ImagePositionPatient') else None,
            is_osirix_sr=hasattr(ds, 'EncapsulatedDocument'))
        ds = Dotdict(ds)
        results.append(ds)
    return results


def build_SOPInstanceUID_lookup_table(dicoms):
    SOPInstanceUID_lookup_table = dict()
    for ds in dicoms:
        SOPInstanceUID_lookup_table[ds.SOPInstanceUID] = ds
    return SOPInstanceUID_lookup_table


def find_osirix_sr(dicoms):
    """
    find OsirixSR files from given list of dicom files
    """
    return [ds for ds in dicoms if ds.is_osirix_sr]


def parse_osirix_sr(osirix_sr: str):
    if not osp.exists(osirix_sr):
        raise FileNotFoundError(f"File {osirix_sr} does not exist.")
    sr = pydicom.dcmread(osirix_sr)
    bytes_data = sr.EncapsulatedDocument
    if bytes_data is None:
        raise ValueError("EncapsulatedDocument is empty")

    try:
        if bool(bytes_data[-1]):
            data = NSKeyedUnArchiver.unserializeNSKeyedArchiver(bytes_data)
        else:
            data = NSKeyedUnArchiver.unserializeNSKeyedArchiver(bytes_data[:-1])
    except Exception as e:
        raise ValueError(f"Failed to unserialize EncapsulatedDocument {sr.fullpath}: {e}")

    rois = Dotdict()
    for d in data:
        n = len(d["points"])

        if n == 0:
            warn(f"Empty points for ROI {d['name']} in {osirix_sr}.")
            continue

        coords = np.array(
            [np.array(list(p.values())[0][1:-1].split(", "), dtype=np.float32) for p in d["points"]]
        ).reshape(-1, 2)

        if d["name"] not in rois:
            rois[d["name"]] = []
        rois[d["name"]].append(coords)
    return rois
