import numpy as np

from dolfinx.mesh import create_unit_square
from dolfinx.fem import FunctionSpace
from dolfinx.geometry import (BoundingBoxTree, compute_colliding_cells,
                              compute_collisions)

from mpi4py import MPI

source_points = np.array([(1/4, 1/4, 0), (1/10, 6/10, 0)])
source_weight = np.array([1.0, 1.0])

mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
tdim  = mesh.topology.dim

# print(mesh.topology.index_map(tdim).size_local)

V = FunctionSpace(mesh, ("Lagrange", 1))

vector = np.zeros(V.dofmap.index_map.size_global)
    
tree = BoundingBoxTree(mesh, mesh.topology.dim)

# print(tree.num_bboxes)

# print(tree)

# print(mesh.geometry.dofmap)

# print(mesh.geometry.x)

# print(mesh.geometry.index_map().size_global)

# print(V.dofmap.index_map.size_global)

# x_g = mesh.geometry.x

# for i in range(mesh.geometry.dofmap.num_nodes):
#     n0, n1, n2 = mesh.geometry.dofmap.links(i)
#     print(f"Cell {i}: {x_g[n0]}, {x_g[n1]}, {x_g[n2]}")

for point, weight in zip(source_points, source_weight):
    
    # Get cell
    cell_candidates = compute_collisions(tree, point)  # return adjacency lists which contains the bounding box index that contains the candidates cell

    # Gets the first cell    
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, point)  # return the cells which collide the point

    # Get the points that define the cell
    # We are using the first colliding cell
    v = [mesh.geometry.x[i] for i in mesh.geometry.dofmap.links(colliding_cells[0])]
    
    # Get local coordinates of the cell
    origin = v[0]
    axes = [vtx - origin for vtx in v[1:]]
    tdim = 3

    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2
        
    assert len(axes) == 3

    # Computes the coefficients of each axes
    local_coordinates = np.linalg.solve(np.array(axes).T, point-origin)[:tdim]

    # Evaluate the basis functions at the local coordinates
    values = V.element.basix_element.tabulate(0, [local_coordinates])[0, 0, :, 0]  # evaluate the basis function at local coordinate points
    dofs = V.dofmap.cell_dofs(colliding_cells[0])  # get the global degrees of freedom of the cell

    # Sum the contribution of the basis functions
    for dof, val  in zip(dofs, values):
        vector[dof] += val * weight


# a = [i for i in vector]
# print(vector)
# print(a)

