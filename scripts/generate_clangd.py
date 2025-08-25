#! /usr/bin/env python3

import os
import yaml
from pathlib import Path

def find_cuda_include_dirs(workspace_root):
    """
    Find all directories containing CUDA header files (.cuh) and 
    directories named 'include' that are in CUDA-related paths.
    """
    include_dirs = set()
    workspace_path = Path(workspace_root)
    
    # Find all directories containing .cuh files
    for cuh_file in workspace_path.rglob("*.cuh"):
        include_dirs.add(str(cuh_file.parent))
    
    # Find all directories containing .cu files (they might have local includes)
    for cu_file in workspace_path.rglob("*.cu"):
        include_dirs.add(str(cu_file.parent))
    
    # Find all 'include' directories in paths that contain 'cuda' or CUDA files
    for include_dir in workspace_path.rglob("include"):
        # Check if this is in a CUDA-related path
        if "cuda" in str(include_dir).lower():
            include_dirs.add(str(include_dir))
        # Also check if there are any .cuh or .cu files in subdirectories
        elif any(include_dir.rglob("*.cuh")) or any(include_dir.rglob("*.cu")):
            include_dirs.add(str(include_dir))
    
    # Remove any paths in target directory (build artifacts)
    include_dirs = {d for d in include_dirs if "/target/" not in d}
    
    return sorted(include_dirs)

def main():
    """
    Generate a .clangd configuration file for the workspace. Note that this should
    be run from the root of the workspace.
    """
    workspace_root = os.getcwd()
    print(f"Generating .clangd for workspace: {workspace_root}")
    
    # Find all CUDA include directories
    cuda_include_dirs = find_cuda_include_dirs(workspace_root)
    
    print(f"Found {len(cuda_include_dirs)} include directories:")
    for dir in cuda_include_dirs:
        rel_path = os.path.relpath(dir, workspace_root)
        print(f"  - {rel_path}")
    
    # Create include flags
    all_includes = [f"-I{dir}" for dir in cuda_include_dirs]
    
    # Add CUDA-specific compile flags
    compile_flags = all_includes + [
        "-x", "cuda",
        "-std=c++17",
        "--cuda-gpu-arch=sm_70",  # Common GPU architecture
        "-D__CUDA_ARCH__=700",     # Define CUDA architecture
    ]
    
    # Final .clangd dictionary
    clangd_config = {
        "CompileFlags": {
            "Add": compile_flags
        },
        "Diagnostics": {
            "UnusedIncludes": "Strict",
            "MissingIncludes": "Strict"
        }
    }
    
    # Write the .clangd file
    output_path = os.path.join(workspace_root, ".clangd")
    with open(output_path, "w") as f:
        yaml.dump(clangd_config, f, sort_keys=False, default_flow_style=False)
    
    print(f"\n✅ .clangd file generated successfully at {output_path}")
    print(f"   Total include directories: {len(cuda_include_dirs)}")
    print(f"   Configuration will apply to all .cu and .cuh files in the repository")

if __name__ == "__main__":
    main()
