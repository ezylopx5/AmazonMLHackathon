# create_submission_package.py
# Package all code and documents for Amazon ML Challenge 2025 submission

import zipfile
import os
from pathlib import Path
import datetime

def create_submission_package():
    """Create a complete submission package with code and documentation"""
    
    print("📦 Creating Amazon ML Challenge 2025 Submission Package")
    print("=" * 60)
    
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"amazon_ml_2025_submission_{timestamp}.zip"
    
    # Files and folders to include
    files_to_include = [
        # Core source code
        "src/__init__.py",
        "src/__main__.py", 
        "src/train_ultra.py",
        "src/train_enhanced.py",
        "src/train_simple.py",
        "src/config.py",
        "src/features.py",
        "src/data.py",
        "src/utils.py",
        "src/models.py",
        
        # Configuration
        "configs/config.yaml",
        
        # Submission and validation scripts
        "create_submission.py",
        "quick_submission.py", 
        "validate.py",
        "check_upload_files.py",
        
        # Documentation
        "SOLUTION_APPROACH_DOCUMENT.md",
        "README.md",
        "requirements.txt",
        
        # Optional: Enhancement guides
        "TOP_50_PLAN.md",
        "UPLOAD_GUIDE.md",
        "SUBMISSION_READY.md",
        
        # Training scripts
        "scripts/train.sh",
        "scripts/predict.sh",
        
        # Pipeline scripts  
        "pipeline.py",
        "preflight_check.py",
    ]
    
    # Optional folders (include if they exist and are small)
    optional_files = [
        "setup.py",
        "error_predictions.py",
        "baseline_submission.py",
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        total_files = 0
        total_size = 0
        
        print("📁 Adding core files...")
        
        # Add main files
        for file_path in files_to_include:
            path = Path(file_path)
            if path.exists():
                zipf.write(path, path)
                size = path.stat().st_size
                total_size += size
                total_files += 1
                print(f"  ✅ {file_path} ({size/1024:.1f}KB)")
            else:
                print(f"  ⚠️ {file_path} (not found)")
        
        print(f"\n📁 Adding optional files...")
        
        # Add optional files
        for file_path in optional_files:
            path = Path(file_path)
            if path.exists():
                zipf.write(path, path)
                size = path.stat().st_size
                total_size += size
                total_files += 1
                print(f"  ✅ {file_path} ({size/1024:.1f}KB)")
        
        # Add sample submission (if it exists)
        submissions_dir = Path("submissions")
        if submissions_dir.exists():
            for sub_file in submissions_dir.glob("*.csv"):
                if sub_file.stat().st_size < 50 * 1024 * 1024:  # Only if < 50MB
                    zipf.write(sub_file, sub_file)
                    size = sub_file.stat().st_size
                    total_size += size
                    total_files += 1
                    print(f"  ✅ {sub_file} ({size/(1024*1024):.1f}MB)")
                    break  # Only include one submission file
    
    print(f"\n" + "=" * 60)
    print(f"📦 SUBMISSION PACKAGE CREATED")
    print(f"=" * 60)
    print(f"📁 Filename: {zip_filename}")
    print(f"📊 Total files: {total_files}")
    print(f"📏 Total size: {total_size/(1024*1024):.2f}MB")
    
    # Verify zip file
    print(f"\n🔍 Package contents:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for info in zipf.infolist():
            print(f"  📄 {info.filename} ({info.file_size} bytes)")
    
    print(f"\n✅ READY FOR SUBMISSION!")
    print(f"📤 Upload: {zip_filename}")
    print(f"📄 Document: SOLUTION_APPROACH_DOCUMENT.md")
    
    return zip_filename

def convert_md_to_pdf():
    """Convert markdown document to PDF (optional)"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from markdown.extensions import codehilite, fenced_code
        
        print("\n📄 Converting approach document to PDF...")
        
        # Read markdown file
        with open("SOLUTION_APPROACH_DOCUMENT.md", "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content, 
            extensions=['codehilite', 'fenced_code', 'tables']
        )
        
        # Add CSS styling
        css = CSS(string='''
            @page { margin: 1in; }
            body { font-family: Arial, sans-serif; line-height: 1.4; }
            h1, h2, h3 { color: #333; }
            code { background-color: #f5f5f5; padding: 2px 4px; }
            pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        ''')
        
        # Generate PDF
        html_doc = HTML(string=f"<html><body>{html_content}</body></html>")
        html_doc.write_pdf("SOLUTION_APPROACH_DOCUMENT.pdf", stylesheets=[css])
        
        print("  ✅ PDF created: SOLUTION_APPROACH_DOCUMENT.pdf")
        
    except ImportError:
        print("  ⚠️ PDF conversion requires: pip install markdown weasyprint")
        print("  📄 Submit the .md file directly")
    except Exception as e:
        print(f"  ❌ PDF conversion failed: {e}")
        print("  📄 Submit the .md file directly")

if __name__ == "__main__":
    # Create submission package
    zip_file = create_submission_package()
    
    # Try to create PDF (optional)
    convert_md_to_pdf()
    
    print(f"\n🎉 SUBMISSION PACKAGE READY!")
    print(f"📦 Code Package: {zip_file}")
    print(f"📄 Approach Document: SOLUTION_APPROACH_DOCUMENT.md")
    print(f"🚀 Ready to submit to Amazon ML Challenge 2025!")