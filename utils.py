# utils.py
import os
import numpy as np
import pandas as pd
import trimesh

def get_mesh(file_path):
    """
    Load a mesh from an OBJ file and return its vertex coordinates.
    Assumes the mesh has a fixed number of vertices (e.g. 3000).
    """
    mesh = trimesh.load(file_path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    return vertices

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


def load_dna_face_data(csv_file, face_folder):
    """
    Load SNP data from a CSV file and corresponding 3D face meshes from OBJ files.
    The CSV must have an 'ID' column and SNP feature columns.
    """
    df = pd.read_csv(csv_file, sep="\t")  # Adjust separator if needed
    faces = []
    snps = []
    ids = []
    
    for _, row in df.iterrows():
        sample_id = row['ID']
        # Extract SNP vector (all columns except 'ID')
        snp_vector = row.drop('ID').values.astype(np.float32)
        # Construct the OBJ filename based on sample ID
        file_name = f"symmetrized_{sample_id}_cleaned.obj"
        file_path = os.path.join(face_folder, file_name)
        # Load the mesh vertices
        mesh_vertices = get_mesh(file_path)
        
        faces.append(mesh_vertices)
        snps.append(snp_vector)
        ids.append(sample_id)
        
    faces = np.array(faces)   # Expected shape: (N, NUM_VERTICES, 3)
    snps = np.array(snps)     # Expected shape: (N, number_of_SNPs)
    return snps, faces, ids

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