# utils.py
import os
import numpy as np
import pandas as pd
import trimesh

def get_mesh_faces(file_path):
    """
    Load a mesh from an OBJ file and return its face connectivity.
    Assumes the mesh has a consistent topology across samples.
    """
    mesh = trimesh.load(file_path, process=False)
    return mesh.faces.astype(np.int32)

def load_fixed_connectivity(template_obj_path):
    """
    Load the face connectivity from a template OBJ file.
    """
    return get_mesh_faces(template_obj_path)

def get_mesh(file_path):
    """
    Load a mesh from an OBJ file and return its vertex coordinates.
    """
    try:
        mesh = trimesh.load(file_path, process=False)
        return mesh.vertices.astype(np.float32)
    except Exception as e:
        print(f"Error loading mesh: {file_path} -> {e}")
        return None  # Return None if loading fails

def load_dna_face_data(csv_file, face_folder):
    """
    Load SNP data from a CSV file and corresponding 3D face meshes from OBJ files.
    Skips entries that do not have a matching 3D face file.
    """
    df = pd.read_csv(csv_file, sep=",")  # Adjust separator if needed
    valid_faces = []
    valid_snps = []
    valid_ids = []

    for _, row in df.iterrows():
        sample_id = row['ID']
        snp_vector = row.drop('ID').values.astype(np.float32)
        file_name = f"symmetrized_{int(sample_id)}_cleaned.obj"
        file_path = os.path.join(face_folder, file_name)

        if os.path.exists(file_path):
            mesh_vertices = get_mesh(file_path)
            if mesh_vertices is not None:
                valid_faces.append(mesh_vertices)
                valid_snps.append(snp_vector)
                valid_ids.append(sample_id)
        else:
            print(f"Skipping sample {sample_id}: No matching OBJ file found.")

    valid_faces = np.array(valid_faces)  # Shape: (N, NUM_VERTICES, 3)
    valid_snps = np.array(valid_snps)    # Shape: (N, num_SNPs)

    print(f"Shapes: {valid_faces.shape}, {valid_snps.shape}")
    
    return valid_snps, valid_faces, valid_ids

def export_mesh_to_obj(vertices, faces, file_path):
    """
    Export a mesh defined by vertices and faces to an OBJ file.
    vertices: numpy array of shape (NUM_VERTICES, 3)
    faces: numpy array of shape (M, 3)
    """
    with open(file_path, "w") as f:
        # Write vertices (formatted to 4 decimal places)
        for v in vertices:
            f.write("v {:.4f} {:.4f} {:.4f}\n".format(v[0], v[1], v[2]))
        # Write faces (OBJ format uses 1-indexed vertices)
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))