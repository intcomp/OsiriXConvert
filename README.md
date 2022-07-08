# OsirixSR Converter

Convert Osirix ROI into standardized formats.

## Overview

The [OsiriX dicom Viewer](https://www.osirix-viewer.com/) 
exports ROI annotations
into the proprietary "OsiriX SR" format
that is hard to parse and read.

This tool parseses data from OsiriX files and converts it into widely compatible formats for easier use and analysis.


### Key Features
- No predefined file structure required; automatically processes studies based on DICOM metadata.
- Supports batch processing.
- Generate multiple output formats:
    - NumPy arrays
    - PNG images
    - [DICOM RT Structure Sets](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.19.html)
    - NIFTI volumes
    - json


## Quick start

```bash
# clone the code
git clone https://github.com/intcomp/OsiriXConvert
cd OsiriXConvert
# install dependencies
pip install -r requirements.txt
# convert
python convert.py /path/to/your/cases
```

You can start with the example `python convert.py /path/to/your/cases`.

Please refer to <https://kaizhao.net/blog/osirix-convert> for detailed
instruction on how to export and convert OsiriX ROI annotations.