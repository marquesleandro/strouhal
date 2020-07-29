# ==========================================
# Code created by Leandro Marques at 12/2018
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# This code is used to compute boundary condition


import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

# OBS: para vetor devemos unir eixo x e y no mesmo vetor, logo usar np.row_stack([dirichlet_pts[1],dirichlet_pts[2]])



class linearStrouhal:
 def __init__(_self, _numPhysical, _numNodes, _x, _y, _Re):
  _self.numPhysical = _numPhysical
  _self.numNodes = _numNodes
  _self.x = _x
  _self.y = _y
  _self.Re = _Re
  _self.inflowVelocity = 1.0
  _self.L = np.max(_self.y)
  _self.benchmark_problem = 'linear Strouhal'


 def xVelocityCondition(_self, _boundaryEdges, _LHS0, _neighborsNodes):
  _self.dirichletVector = np.zeros([_self.numNodes,1], dtype = float) 
  _self.dirichletNodes = [] 
  _self.aux1BC = np.zeros([_self.numNodes,1], dtype = float) #For scipy array solve
  _self.aux2BC = np.ones([_self.numNodes,1], dtype = float) 
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.boundaryEdges = _boundaryEdges
  _self.neighborsNodes = _neighborsNodes

 # Dirichlet condition
  for i in range(0, len(_self.boundaryEdges)):
   line = _self.boundaryEdges[i][0]
   v1 = _self.boundaryEdges[i][1] - 1
   v2 = _self.boundaryEdges[i][2] - 1

   # Noslip 
   if line == 5:
    _self.aux1BC[v1] = 0.0
    _self.aux1BC[v2] = 0.0
 
    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

   # Inflow
   elif line == 4:
    _self.aux1BC[v1] = _self.inflowVelocity
    _self.aux1BC[v2] = _self.inflowVelocity

    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

  _self.dirichletNodes = np.unique(_self.dirichletNodes)


  # Gaussian elimination for vx
  for mm in _self.dirichletNodes:
   for nn in _self.neighborsNodes[mm]:
    _self.dirichletVector[nn] -= float(_self.LHS[nn,mm]*_self.aux1BC[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.dirichletVector[mm] = _self.aux1BC[mm]
   _self.aux2BC[mm] = 0.0
 



 def yVelocityCondition(_self, _boundaryEdges, _LHS0, _neighborsNodes):
  _self.dirichletVector = np.zeros([_self.numNodes,1], dtype = float) 
  _self.dirichletNodes = [] 
  _self.aux1BC = np.zeros([_self.numNodes,1], dtype = float) #For scipy array solve
  _self.aux2BC = np.ones([_self.numNodes,1], dtype = float) 
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.boundaryEdges = _boundaryEdges
  _self.neighborsNodes = _neighborsNodes

 # Dirichlet condition
  for i in range(0, len(_self.boundaryEdges)):
   line = _self.boundaryEdges[i][0]
   v1 = _self.boundaryEdges[i][1] - 1
   v2 = _self.boundaryEdges[i][2] - 1

   # Noslip 
   if line == 1 or line == 3 or line == 4 or line == 5:
    _self.aux1BC[v1] = 0.0
    _self.aux1BC[v2] = 0.0
 
    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

  _self.dirichletNodes = np.unique(_self.dirichletNodes)


  # Gaussian elimination for vy
  for mm in _self.dirichletNodes:
   for nn in _self.neighborsNodes[mm]:
    _self.dirichletVector[nn] -= float(_self.LHS[nn,mm]*_self.aux1BC[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.dirichletVector[mm] = _self.aux1BC[mm]
   _self.aux2BC[mm] = 0.0
 


 def streamFunctionCondition(_self, _boundaryEdges, _LHS0, _neighborsNodes):
  _self.dirichletVector = np.zeros([_self.numNodes,1], dtype = float) 
  _self.dirichletNodes = [] 
  _self.aux1BC = np.zeros([_self.numNodes,1], dtype = float) #For scipy array solve
  _self.aux2BC = np.ones([_self.numNodes,1], dtype = float) 
  _self.LHS = sps.csr_matrix.copy(_LHS0) #used csr matrix because LHS = lil_matrix + lil_matrix
  _self.boundaryEdges = _boundaryEdges
  _self.neighborsNodes = _neighborsNodes

 # Dirichlet condition
  for i in range(0, len(_self.boundaryEdges)):
   line = _self.boundaryEdges[i][0]
   v1 = _self.boundaryEdges[i][1] - 1
   v2 = _self.boundaryEdges[i][2] - 1

   # Bottom Line
   # psi_bottom can be any value. Because, important is psi_top - psi_bottom.
   # In this case, psi_bottom is zero
   if line == 1:
    _self.aux1BC[v1] = 0.0
    _self.aux1BC[v2] = 0.0
 
    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

   # Top Line
   # Ref: Batchelor 1967 pag. 76 eq. 2.2.8
   elif line == 3:
    _self.aux1BC[v1] = _self.L*_self.inflowVelocity
    _self.aux1BC[v2] = _self.L*_self.inflowVelocity

    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

   # Circle
   elif line == 5:
    _self.aux1BC[v1] = (_self.L*_self.inflowVelocity)/2.0
    _self.aux1BC[v2] = (_self.L*_self.inflowVelocity)/2.0

    _self.dirichletNodes.append(v1)
    _self.dirichletNodes.append(v2)

  _self.dirichletNodes = np.unique(_self.dirichletNodes)


  # Gaussian elimination for psi
  for mm in _self.dirichletNodes:
   for nn in _self.neighborsNodes[mm]:
    _self.dirichletVector[nn] -= float(_self.LHS[nn,mm]*_self.aux1BC[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.dirichletVector[mm] = _self.aux1BC[mm]
   _self.aux2BC[mm] = 0.0
 


 def pressureCondition(_self, _boundaryEdges, _LHS0, _neighborsNodes):
  _self.dirichletVector = np.zeros([_self.numNodes,1], dtype = float) 
  _self.dirichletNodes = [] 
  _self.aux1BC = np.zeros([_self.numNodes,1], dtype = float) #For scipy array solve
  _self.aux2BC = np.ones([_self.numNodes,1], dtype = float) 
  _self.LHS = sps.csr_matrix.copy(_LHS0) #used csr matrix because LHS = lil_matrix + lil_matrix
  _self.boundaryEdges = _boundaryEdges
  _self.neighborsNodes = _neighborsNodes

  # Dirichlet condition
  _self.aux1BC[0] = 0.0 # node (0,0) null pressure
  _self.dirichletNodes.append(0)
  _self.dirichletNodes = np.unique(_self.dirichletNodes)

  # Gaussian elimination for pressure
  for mm in _self.dirichletNodes:
   for nn in _self.neighborsNodes[mm]:
    _self.dirichletVector[nn] -= float(_self.LHS[nn,mm]*_self.aux1BC[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.dirichletVector[mm] = _self.aux1BC[mm]
   _self.aux2BC[mm] = 0.0
 



