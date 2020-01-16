import wavefront_reader
import numpy as np
from sklearn.decomposition import PCA


def arena_obj_to_transformation(objfile, object="Arena"):
    """Returns the 3x3 rotation matrix and 1x3 translation array for the "object" object in the objfile, using PCA."""
    verts = wavefront_reader.read_objfile(objfile)[object]['v']
    uverts = np.unique(verts, axis=0)
    floor_verts = uverts[uverts[:,1].argsort()][:4, :]
    pca = PCA(n_components=3).fit(floor_verts)
    rotmat, translation = pca.components_, pca.mean_
    return rotmat, translation



def make_arena_coords_obj(rotmat, translation):
    """Should (untested!!!) return a ratcave object with the transformation values applied."""
    import ratcave as rc
    mesh = rc.EmptyMesh()
    mesh.position.xyz = translation
    mesh.rotation = rc.RotationEuler.from_matrix(rotmat)\
    return mesh



# Example usage below (maybe needs some tweaking--untested):
if __name__ == "__main__":
    import ratcave as rc
    objfile = "arena.obj"
    arena = rc.read_wavefront(objfile).get_mesh("Arena")
    
    rotmat, translation = arena_obj_to_transformation(objfile)
    exp_transform = make_arena_coords_obj(rotmat, translation)
    exp_transform.parent = arena
    
    
    
