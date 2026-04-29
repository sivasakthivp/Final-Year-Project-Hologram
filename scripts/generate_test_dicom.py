"""
Generate Sample DICOM Files for Testing
Creates synthetic medical imaging data for Streamlit frontend testing
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Try to import pydicom; install if needed
try:
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
except ImportError:
    print("❌ pydicom not installed. Installing...")
    os.system("pip install pydicom")
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian


def create_ct_dicom(output_path, patient_name="TEST^PATIENT", patient_id="12345"):
    """
    Create a synthetic CT DICOM file
    
    Args:
        output_path: Path where to save the DICOM file
        patient_name: Patient name in DICOM format (LastName^FirstName)
        patient_id: Patient ID
    """
    
    # Create file metadata
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9'
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Create dataset
    ds = FileDataset(output_path, {}, preamble=b"\0" * 128, file_meta=file_meta)
    
    # Patient info
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = '19700101'
    ds.PatientAge = '050Y'
    ds.PatientSex = 'M'
    
    # Study info
    ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9'
    ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.1'
    ds.SOPInstanceUID = '1.2.3.4.5.6.7.8.9.1.1'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.2'
    
    # Study/Series dates
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    
    # Modality
    ds.Modality = 'CT'
    ds.SeriesDescription = 'CT Head'
    ds.StudyDescription = 'Head CT Scan'
    
    # Image info
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    
    # Create realistic CT-like pixel data
    # Simulate a head scan with some anatomical structure
    np.random.seed(42)
    
    # Background (air/dark)
    pixel_array = np.random.randint(0, 100, (512, 512), dtype=np.uint16)
    
    # Create circular head-like structure
    center_x, center_y = 256, 256
    radius = 150
    
    for i in range(512):
        for j in range(512):
            dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            
            if dist < radius:
                if dist < 50:  # Inner structure (ventricles/brain)
                    pixel_array[i, j] = np.random.randint(600, 800)
                elif dist < 100:  # Brain matter
                    pixel_array[i, j] = np.random.randint(400, 600)
                else:  # Outer (skull/bone)
                    pixel_array[i, j] = np.random.randint(800, 1000)
    
    # Add some edge features
    pixel_array[200:300, 200] = 950  # Artificial edge
    pixel_array[200, 200:300] = 950
    
    ds.PixelData = pixel_array.tobytes()
    
    # Additional required fields
    ds.ReferringPhysicianName = 'DR^TEST'
    ds.InstitutionName = 'Test Hospital'
    ds.ManufacturerModelName = 'TestScanner'
    ds.Manufacturer = 'TestMfg'
    
    # Save DICOM file
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)
    
    print(f"✅ Created: {output_path}")
    return output_path


def create_mri_dicom(output_path, patient_name="TEST^MRI", patient_id="12346"):
    """Create a synthetic MRI DICOM file"""
    
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.4'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9'
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(output_path, {}, preamble=b"\0" * 128, file_meta=file_meta)
    
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = '19750515'
    ds.PatientAge = '048Y'
    ds.PatientSex = 'F'
    
    ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9.2'
    ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.2.1'
    ds.SOPInstanceUID = '1.2.3.4.5.6.7.8.9.2.1.1'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.4'
    
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    
    ds.Modality = 'MR'
    ds.SeriesDescription = 'MRI Spine T2'
    ds.StudyDescription = 'Spine MRI'
    
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    
    # MRI-like pixel data (different contrast)
    np.random.seed(123)
    pixel_array = np.random.randint(200, 400, (512, 512), dtype=np.uint16)
    
    # Add spine-like structure
    pixel_array[200:300, 240:268] = np.random.randint(600, 800)  # Vertebrae
    
    # Add some anatomical variation
    for i in range(512):
        pixel_array[i, 256 + int(20 * np.sin(i / 50))] = 900
    
    ds.PixelData = pixel_array.tobytes()
    
    ds.ReferringPhysicianName = 'DR^MRI'
    ds.InstitutionName = 'Test Medical Center'
    ds.ManufacturerModelName = 'TestMRI'
    ds.Manufacturer = 'TestMfg'
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)
    
    print(f"✅ Created: {output_path}")
    return output_path


def create_xray_dicom(output_path, patient_name="TEST^XRAY", patient_id="12347"):
    """Create a synthetic X-Ray DICOM file"""
    
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'  # CR Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9'
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(output_path, {}, preamble=b"\0" * 128, file_meta=file_meta)
    
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = '19650320'
    ds.PatientAge = '058Y'
    ds.PatientSex = 'M'
    
    ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9.3'
    ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.3.1'
    ds.SOPInstanceUID = '1.2.3.4.5.6.7.8.9.3.1.1'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'
    
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesDate = datetime.now().strftime('%Y%m%d')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.SeriesTime = datetime.now().strftime('%H%M%S')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    
    ds.Modality = 'CR'
    ds.SeriesDescription = 'CR Chest'
    ds.StudyDescription = 'Chest X-Ray'
    
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 1024
    ds.Columns = 1024
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    
    # X-Ray-like pixel data (chest radiograph simulation)
    np.random.seed(456)
    pixel_array = np.random.randint(100, 200, (1024, 1024), dtype=np.uint16)
    
    # Simulate lung fields
    pixel_array[200:800, 100:500] = np.random.randint(1000, 3000)  # Left lung
    pixel_array[200:800, 524:900] = np.random.randint(1000, 3000)  # Right lung
    
    # Simulate heart silhouette
    pixel_array[300:700, 450:550] = np.random.randint(4000, 5000)
    
    # Simulate ribs
    for y in range(200, 800, 80):
        pixel_array[y:y+30, 100:900] = np.maximum(pixel_array[y:y+30, 100:900], 6000)
    
    ds.PixelData = pixel_array.tobytes()
    
    ds.ReferringPhysicianName = 'DR^XRAY'
    ds.InstitutionName = 'Radiology Department'
    ds.ManufacturerModelName = 'TestXRay'
    ds.Manufacturer = 'TestMfg'
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)
    
    print(f"✅ Created: {output_path}")
    return output_path


def main():
    """Generate sample DICOM files"""
    
    # Create test directory
    test_dir = Path(__file__).parent.parent / "data" / "test_samples"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("  Sample DICOM Generator")
    print("="*50 + "\n")
    
    # Generate different modalities
    files_created = []
    
    print("📋 Generating test DICOM files...\n")
    
    # CT Scan
    ct_file = test_dir / "test_ct_head.dcm"
    create_ct_dicom(str(ct_file), patient_name="SMITH^JOHN", patient_id="CT001")
    files_created.append(ct_file)
    
    # MRI Scan
    mri_file = test_dir / "test_mri_spine.dcm"
    create_mri_dicom(str(mri_file), patient_name="JOHNSON^JANE", patient_id="MRI001")
    files_created.append(mri_file)
    
    # X-Ray
    xray_file = test_dir / "test_xray_chest.dcm"
    create_xray_dicom(str(xray_file), patient_name="BROWN^ROBERT", patient_id="XR001")
    files_created.append(xray_file)
    
    print("\n" + "="*50)
    print("✅ Test samples created successfully!\n")
    
    for f in files_created:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  📁 {f.name}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Path: {f}\n")
    
    print("="*50)
    print("\n🚀 Ready for testing!\n")
    print("Upload these files to test the Streamlit app:")
    print(f"  Location: {test_dir}\n")
    
    return files_created


if __name__ == "__main__":
    main()
