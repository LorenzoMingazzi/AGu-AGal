from fenics import *
import mshr as mr
from mpi4py import MPI
import shutil
import numpy as np
from scipy import spatial
import ufl as uf
import meshio
import sys, os, sympy, shutil, math

import datetime

from math import sqrt

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

outputFolder = "Esempio_import_h2l/"

if mpi_rank == 0:
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    os.makedirs(outputFolder)

# --- Setup (man ffc)
parameters["form_compiler"]["optimize"]      = True
parameters["form_compiler"]["cpp_optimize"]  = True
parameters["form_compiler"]["representation"]= "uflacs"
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# ---Input data
print("Mesh")
L = 10.; H = 3.;
y_bot1 = 0.9999; y_bot2 = 1     #Parametri che modificano l'altezza della zona BOT che si rompe
y_top1 = 2; y_top2 = 2.0001

# --- Mesh
mesh_import = meshio.read("dominio_3.msh")
punti = mesh_import.points
celle = mesh_import.cells
mesh_out = meshio.Mesh(punti, celle)
meshio.write("mesh_out.xml",mesh_out)
mesh = Mesh("mesh_out.xml")
ndim = mesh.topology().dim()

File(outputFolder + "/mesh.pvd") << mesh

"""
Struttura layer
_____________________________________________________________
|                      zona che danneggia                   |
|                                                           |
|--------------------------y_top2---------------------------|
|                zona transizione resistenza                |
|--------------------------y_top1---------------------------|
|                                                           |
|                 core che non danneggia                    |
|                                                           |
|--------------------------y_bot2---------------------------|
|                 zona transizione resistenza               |
|--------------------------y_bot1---------------------------|
|                                                           |
|_____________________zona che danneggia____________________|

"""

# ---Material parameter
print("Material")

E1 = 7000.
nu1 = 0.2
ell1 = 0.12
w1 = 1.428e-4
Gc1 = 5.497e-6
sigma1 = (w1*E1)**0.5

E2 = 50000.
nu2 = 0.2
w2 = 2e-3
Gc2 = 1.886
ell2 = 0.05
sigma2 = (w2*E2)**0.5

kres_1 = Constant(1e-10)

chi_E = Constant(E2/E1)
chi_Gc = Constant(Gc2/Gc1)
chi_w = Constant(w2/w1)
chi_ell = Constant(ell2/ell1)
chi_s = Constant(sigma2/sigma1)

# --- Mesh
maxiter = 2000

# --- cell_size, ndim, tree
ref_in = 4
ref_add = 2
ref_active = ref_in-ref_add

cell_size = Cell(mesh,0).h()

G = []
for ii in cells(mesh):
    G.append([ii.midpoint().x(), ii.midpoint().y()])
tree = spatial.cKDTree(G)

mf_test = MeshFunction('size_t', mesh, 2, 0)
mf_test.array()[0] = 1
mesh_sub = SubMesh(mesh, mf_test, 1)
i = 0
while i < ref_in:
    mesh_sub_ref = refine(mesh_sub)
    mesh_sub = mesh_sub_ref
    i += 1
fine_cell_size = Cell(mesh_sub,0).h()

print('Cell size = ', cell_size)
print('Fine Cell size = ', fine_cell_size)

# --- Modifica numero cicli di raffittimento
print('Refinement Parameter')
beta_1, beta_2, beta_3, alpha_soglia = 1, 1.1, 1.0, 0.9
En_rif = beta_1 * (w1 / (2**0.5))
coeff = beta_2 * cell_size
coeff_mark_en = beta_3 * cell_size
alpha_s = alpha_soglia

# --- Function Spaces
print("Function Spaces")

# ---Displacement
V = VectorElement("Lagrange", mesh.ufl_cell(), 1)
V_u = FunctionSpace(mesh, V)

u = Function(V_u)
du = TrialFunction(V_u)
v = TestFunction(V_u)
u.rename('u', 'u')

# ---Damage
V_alpha = FunctionSpace(mesh, "CG", 1)

alpha = Function(V_alpha)
dalpha = TrialFunction(V_alpha)
beta = TestFunction(V_alpha)
alpha.rename('damage', 'damage')
alpha_0 = Function(V_alpha)

alpha_lb = interpolate(Constant("0."), V_alpha)
alpha_ub = interpolate(Constant("1."), V_alpha)

# --- Flags
V_Source = FunctionSpace(mesh, 'DG', 0)
V_flag = FunctionSpace(mesh, "DG", 0)
alpha_DG = Function(V_flag)

# ---Boundaries
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H)
top = Top()

class Bot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0)
bot = Bot()

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)
right = Right()

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)
left = Left()

class R_top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and x[1] >= y_top2
r_top = R_top()

class R_mid(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and between(x[1], (y_bot1, y_top2))
r_mid = R_mid()

class R_bot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and x[1] <= y_bot1
r_bot = R_bot()

class Core(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (y_bot1, y_top2))
core = Core()

class Out(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < y_bot1 and x[1] > y_top2
out = Out()

def Node_M(x):
    return near(x[0], L/2.) and near(x[1], 0)

# ---Measures
domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
domains.set_all(0)
core.mark(domains, 1)
dom_refine = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bot.mark(boundaries, 4)

dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ---Displacement BC
print("Boundary Conditions")

u_D = Expression('t', t = 0, degree=0)
u_R = Expression('t', t = 0, degree=0)
u_L = Expression('t', t = 0, degree=0)

bcu_l = DirichletBC(V_u.sub(0), Constant(0.), boundaries, 1) # left
bcu_r = DirichletBC(V_u.sub(0), u_D, boundaries, 3)          # right
bcu_b = DirichletBC(V_u.sub(1), Constant(0.), boundaries, 4) # bot
bcu = [bcu_l, bcu_r, bcu_b]

# ---Damage BC
bca = [DirichletBC(V_alpha, Constant(0.), boundaries, 1),
       DirichletBC(V_alpha, Constant(0.), boundaries, 3)]

# ---Definitions
print("Definitions")
def eps(u):
    return sym(grad(u))

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha

def a(alpha,kres):
    """Stiffness modulation (degradation function) as a function of damage"""
    return (1 - alpha) ** 2. + kres

def sigma_0(u, E, nu):
    mu = E / (2. * (1. + nu))
    lamb = E * nu / (1. - nu ** 2)
    return 2. * mu * (eps(u)) + lamb * tr(eps(u)) * Identity(3)

def sigma(u, alpha, E, nu, kres):
    return (a(alpha, kres)) * sigma_0(u, E, nu)

z = sympy.Symbol("z")
c_w = 4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1))

# ---Loading
print("Loading")

load0 = (w1*E1)**0.5 /E1
loads = load0 * np.linspace(0., 15, 61)

# ---UFL Forms
print("UFL forms")

elastic_energy = 0.5 * inner(sigma(u, alpha, E1, nu1, kres_1), eps(u))*dx(0)(metadata={'quadrature_degree': 5}) +\
                 0.5 * inner(sigma(u, alpha, E2, nu2, kres_1), eps(u))*dx(1)(metadata={'quadrature_degree': 5})
dissipated_energy = Gc1 / float(c_w) * (w(alpha) / ell1
                    + ell1 * dot(grad(alpha), grad(alpha))) * dx(0) + \
                    Gc2 / float(c_w) * (w(alpha) / ell1
                    + ell1 * dot(grad(alpha), grad(alpha))) * dx(1)
total_energy = elastic_energy + dissipated_energy

# First directional derivative wrt u
E_u = derivative(total_energy, u, v)
E_du = uf.replace(E_u, {u: du})

# First and second directional derivative wrt alpha
E_da = derivative(total_energy, alpha, beta)
E_dda = derivative(E_da, alpha, dalpha)

# ---Problem
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bcu)
solver_u = LinearVariationalSolver(problem_u)

# --- Setup savefiles
energies = np.zeros((len(loads), 4))
energies_r = np.zeros((len(loads), 4))
forces = np.zeros((len(loads), 2))

# --- Displacement savefile
File_u = File(outputFolder + "/u.pvd")
File_ur = File(outputFolder + "/u_r.pvd")

# --- Damage savefile
File_d = File(outputFolder + "/d.pvd")
File_dr = File(outputFolder + "/dr.pvd")
File_dr_it = File(outputFolder + "/dr_it.pvd")

File_mark = File(outputFolder + "/d_en.pvd")

def postprocessing():
    if (flag_ma > 0):
        File_ur << (ur, t)
        File_dr << (alpha_r, i_t * 1000 + alternate_iter)
        elastic_energy_value = assemble(elastic_energy_r)
        surface_energy_value = assemble(dissipated_energy_r)

        boundaries_r_f = MeshFunction('size_t', mesh1_ref, 1, 0)
        r_top.mark(boundaries_r_f, 1)
        r_bot.mark(boundaries_r_f, 1)
        r_mid.mark(boundaries_r_f, 2)
        ds_r_f = Measure('ds', domain=mesh1_ref, subdomain_data=boundaries_r_f)
        Fx = assemble(sigma(ur, alpha_r, E1, nu1, kres_1)[0,0]*ds_r_f(1)) + \
             assemble(sigma(ur, alpha_r, E2, nu2, kres_1)[0,0]*ds_r_f(2))

    else:
        File_u << (u, t)
        File_d << (alpha, t)
        elastic_energy_value = assemble(elastic_energy)
        surface_energy_value = assemble(dissipated_energy)

        boundaries_f = MeshFunction('size_t', mesh, 1, 0)
        ds_f = Measure('ds', domain=mesh, subdomain_data=boundaries_f)
        r_top.mark(boundaries_f, 1)
        r_bot.mark(boundaries_f, 1)
        r_mid.mark(boundaries_f, 2)
        Fx = assemble(sigma(u, alpha, E1, nu1, kres_1)[0,0]*ds_f(1)) +\
             assemble(sigma(u, alpha, E2, nu2, kres_1)[0,0]*ds_f(2))

    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value + surface_energy_value])
    forces[i_t] = np.array([t, Fx])

    np.savetxt(outputFolder + '/energies.txt', energies)
    np.savetxt(outputFolder + '/forces.txt', forces)


# --- Useful functions
def list_transport(list_i, mesh, mf_type):
    """Save the ist of elements over the refined mesh and return the new list and the new meshfunction"""
    list_mark = list(map(int, list_i))
    if mf_type == 'size_t':
        mesh_func_name = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        mesh_func_name.array()[list_mark] = 1
        if len(np.argwhere(mesh_func_name.array() == 1)) > 0:
            new_elem_list = np.concatenate(np.argwhere(mesh_func_name.array() == 1))
        else:
            new_elem_list = []
    else:
        mesh_func_name = MeshFunction('bool', mesh, mesh.topology().dim(), False)
        mesh_func_name.array()[list_mark] = True
        if len(np.argwhere(mesh_func_name.array() == True)) > 0:
            new_elem_list = np.concatenate(np.argwhere(mesh_func_name.array() == True))
        else:
            new_elem_list = []
    return new_elem_list, mesh_func_name

def list_transport_for(list_to_search, mesh, mf_type, rmap):
    """Recover the child elements after the refine within the for loop"""
    elem_list_mid = np.where(np.in1d(rmap, list_to_search))[0]
    list_to_search = elem_list_mid
    elem_list_mid = []
    if mf_type == 'bool':
        mf_new = MeshFunction('bool', mesh, mesh.topology().dim(), False)
        mf_new.array()[list_to_search] = True
    else:
        mf_new = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        mf_new.array()[list_to_search] = 1
    return list_to_search, mf_new

# --- Flags and list initialization
flag_ma = 0
flag_ref = 0
flag_tmp = 0
flag_mesh3 = 0
flag_search = 0
flag_apply = 0
flag_refine = 0
flag_final_ref = 0
flag_mark_alpha = 0
flag_prima_cond = 0
flag_load = 0
flag_load_ref = 0
flag_already_ref = 0

mark_en_save = [1.]
mark_alpha_save = [1.]
mark_alpha_save_it = [1.]

mark_en_no_ref_old = []
mark_en_old = []
mark_en_oold = []
mark_en_ooold = []
mark_en_oooold = []

d_en_back = []
d_en_bback = []
d_en_bbback = []
d_en_bbbback = []

cell_to_search = []
last_cell_refined = []
mark_alpha_to_remove = []
mark_alpha = []
mark_alpha_oldTs = []
mark_alpha_already_rechecked = []
mark_alpha_to_add = []
mark_alpha_coarse = []
mark_alpha_coarse_old = []

mark_add = []

mark_alpha_for_ref = []
ma_for_ref = []
ma_oldTs_base = []

alternate_conv_tol = 2.e-5
G_alpha = []

maxiter = 1000

for (i_t, t) in enumerate(loads):

    if flag_load == 0:
        u_D.t = t
        # u_L.t = -0.5 * t
        # u_R.t = 0.5 * t
        load_save = t
        load_incr = 0
    else:
        # u_D.t = load_save
        u_D.t = t
        # u_L.t = -0.5 * t
        # u_R.t = 0.5 * t

    if t == loads[-1]:
        flag_final_ref = 1

    if list(mark_alpha) != list(mark_alpha_oldTs):
        print('Mark_alpha -> liste DIVERSE')
        flag_ref = 1
        mark_alpha_ooldTs = mark_alpha_oldTs
        mark_alpha_oldTs = mark_alpha

    alternate_convergence = False
    alternate_iter = 0
    alpha_0.assign(alpha)
    err_alpha3_norm = 1

    mark_en = []
    mark_en_prev_iter = []
    mark_diff_save = []
    mark_add = []
    mark_add_prev_iter = []
    mark_add_diff_save = []

    while alternate_convergence is not True \
     and alternate_iter < maxiter:

        alternate_iter += 1

        if mpi_rank == 0:
            print(30 * '-' + " alternate_iter: {}".format(alternate_iter))

        if flag_ma == 0:
            solver_u.solve()

            dom_source = MeshFunction('size_t', mesh, 2, 0)
            dom_source.set_all(0)
            core.mark(dom_source, 1)
            dom_mask = np.concatenate(np.argwhere(dom_source.array() == 1))

            source = inner(sigma_0(u, E1, nu1), eps(u))
            source_projected = project(source, V_Source)
            d_en = (En_rif - source_projected.vector()[:])
            d_en[dom_mask] = 1e-11

        else:
            dom_source = MeshFunction('size_t', mesh1_ref, 2, 0)
            dom_source.set_all(0)
            core.mark(dom_source, 1)
            dom_mask = np.concatenate(np.argwhere(dom_source.array() == 1))

            source_r = inner(sigma_0(ur, E1, nu1), eps(ur))
            V_en_ref = FunctionSpace(mesh1_ref, 'DG', 0)
            source_projected = project(source_r, V_en_ref)
            d_en_r = (En_rif - source_projected.vector()[:])
            d_en_r[dom_mask] = 1e-11

#------------------------------------------------------------------------------

        if d_en.min() < 0:
            print('d_en.min < 0')

            if True:
                if flag_ma > 0:
                    LagrangeInterpolator.interpolate(u, ur)
                dom_source = MeshFunction('size_t', mesh, 2, 0)
                dom_source.set_all(0)
                core.mark(dom_source, 1)
                dom_mask = np.concatenate(np.argwhere(dom_source.array() == 1))

                source = inner(sigma_0(u, E1, nu1), eps(u))
                source_projected = project(source, V_Source)
                d_en = (En_rif - source_projected.vector()[:])
                d_en[dom_mask] = 1e-11
                mark_en = np.concatenate(np.argwhere(d_en < 0))
            else:
                if flag_ma == 0:
                    mark_en_prev_iter = mark_en
                    mark_en = np.concatenate(np.argwhere(d_en < 0))
                else:
                    mark_en_r = np.concatenate(np.argwhere(d_en_r < 0))
                    for ll in range(len(rmap_all)):
                        rmap_array = np.array(rmap_all[len(rmap_all) - 1 - ll])
                        mark_en_coarse = rmap_array[mark_en_r]
                        mark_en_r = mark_en_coarse
                    mark_en_prev_iter = mark_en
                    mark_en = mark_en_coarse

            if flag_refine > 0:
                if flag_final_ref == 0:
                    mark_en_no_ref = np.concatenate((mark_en_oooold, mark_en_ooold, mark_en_oold, mark_en_old), axis = None)
                    mark_en_search = list(set(mark_en_no_ref) - set(mark_en_oooold))
                    
                else:
                    mark_en_no_ref = np.concatenate((mark_en_oooold, mark_en_ooold, mark_en_oold, mark_en_old), axis = None)
                    mark_en_search = list(set(mark_en))
            else:
                mark_en_no_ref = []
                mark_en_search = []

            mark_en_ref = list(set(mark_en) - set(mark_en_no_ref)) #zona che avanza dove refinisco aggro

            # Aggiungo elemento che c'e di differenza tra il passo attuale e quello precedente per evitare
            # il caso in cui ho un elemento che viene aggiunto e tolto di continuo
            # mark_diff_last_iter = list(set(mark_en_prev_iter) - set(mark_en))
            # if len(mark_diff_last_iter)>0:
            #     mark_diff_save = list(set(np.concatenate((mark_diff_save, mark_diff_last_iter), axis = None)))
            # mark_en_ref = np.concatenate((mark_en_ref, mark_diff_save), axis = None)

            source_projected.rename('source','source')
            mf_plot = MeshFunction('size_t', mesh, 2, 0)
            mf_plot.array()[mark_en] = 2
            mark_en_ref_plot = list(map(int, mark_en_ref))
            mf_plot.array()[mark_en_ref_plot] = 1
            File_mark << (mf_plot, i_t * 1000 + alternate_iter)

            if list(mark_en) == list(mark_en_save) and flag_final_ref == 0 and flag_load_ref == 0:
                print('UGUALE')

                problem_ur = LinearVariationalProblem(lhs(E_dur), rhs(E_dur), ur, bcu_r)
                solver_ur = LinearVariationalSolver(problem_ur)

                alpha_r0.assign(alpha_r)

                problem_alpha_r = NonlinearVariationalProblem(E_dar, alpha_r, bcar, E_ddar)
                solver_alpha_r  = NonlinearVariationalSolver(problem_alpha_r)
                problem_alpha_r.set_bounds(alpha_r_lb, alpha_r_ub)

                # Solver for the alpha-problem
                pra = solver_alpha_r.parameters
                pra['nonlinear_solver'] = 'snes'
                pra["snes_solver"]["method"] = "vinewtonrsls"
                pra["snes_solver"]["maximum_iterations"] = 300
                pra["snes_solver"]["absolute_tolerance"] = 1E-7
                pra["snes_solver"]["relative_tolerance"] = 1E-7
                pra["snes_solver"]["linear_solver"] = "mumps"
                pra["snes_solver"]["line_search"] = "basic"
                pra["snes_solver"]["report"] = True

            else:
                print('DIVERSO')
                mark_add_prev_iter = mark_add
                mark_add = []
                for el_id in mark_en:
                    p_x = Cell(mesh, el_id).midpoint().x()
                    p_y = Cell(mesh, el_id).midpoint().y()
                    pp = [p_x, p_y]
                    mark_add.append(tree.query_ball_point(pp, coeff_mark_en, np.float('inf')))
                mark_add = list(set(np.concatenate(mark_add)))

                # print(mark_add_diff_save)
                # mark_add_diff_last_iter = list(set(mark_add_prev_iter) - set(mark_add))
                # if len(mark_add_diff_last_iter)>0:
                #     mark_add_diff_save = list(set(np.concatenate((mark_add_diff_save, mark_add_diff_last_iter), axis = None)))
                # mark_add = np.concatenate((mark_add, mark_add_diff_save), axis = None)

                mark_en_add= np.concatenate([mark_en, mark_add])

#------------------------------------------------------------------------------

                print('Refine mark_en')
                mesh_mid = mesh
                test_mf_en_add, mf_en_add = list_transport(mark_add, mesh_mid, 'bool')
                test_mf_en, mf_en = list_transport(mark_en_ref, mesh_mid, 'bool')

                if len(mark_en_no_ref) > 0:
                    test_mf_en_no_ref_mesh, mf_en_no_ref_mesh = list_transport(mark_en_no_ref, mesh_mid, 'size_t')

                if len(mark_en_search) > 0:
                    test_mf_en_search_mesh, mf_en_search_mesh = list_transport(mark_en_search, mesh_mid, 'size_t')

                rmap_all = []
                for i in range(ref_in):
                    if i < ref_add:
                        mesh_mid_ref = refine(mesh_mid, mf_en_add)
                    else:
                        mesh_mid_ref = refine(mesh_mid, mf_en)

                    rmap = mesh_mid_ref.data().array("parent_cell", ndim)
                    rmap_all.append(rmap)
                    if i < ref_add:
                        test_mf_en_add, mf_en_add = list_transport_for(test_mf_en_add, mesh_mid_ref, 'bool', rmap)
                        test_mf_en, mf_en = list_transport_for(test_mf_en, mesh_mid_ref, 'bool', rmap)
                        if len(mark_en_no_ref) > 0:
                            test_mf_en_no_ref_mesh, mf_en_no_ref_mesh = list_transport_for(test_mf_en_no_ref_mesh, mesh_mid_ref, 'size_t', rmap)
                        if len(mark_en_search) > 0:
                            test_mf_en_search_mesh, mf_en_search_mesh = list_transport_for(test_mf_en_search_mesh, mesh_mid_ref, 'size_t', rmap)
                    else:
                        test_mf_en, mf_en = list_transport_for(test_mf_en, mesh_mid_ref, 'bool', rmap)
                        if len(mark_en_no_ref) > 0:
                            test_mf_en_no_ref_mesh, mf_en_no_ref_mesh = list_transport_for(test_mf_en_no_ref_mesh, mesh_mid_ref, 'size_t', rmap)
                        if len(mark_en_search) > 0:
                            test_mf_en_search_mesh, mf_en_search_mesh = list_transport_for(test_mf_en_search_mesh, mesh_mid_ref, 'size_t', rmap)
                    mesh_mid = mesh_mid_ref

#----------------------------------------------

                print('alpha_s refine')
                if len(mark_en_no_ref) > 0: # ho elementi da raffittire specifici nella strisciolina danneggiata
                    for sub_cicles in range(ref_active):
                        if (list(mark_en_search) != list(mark_search_save) or list(mark_alpha_save_it) != list(mark_alpha_save)) and flag_search == 0 or flag_final_ref  == 1 or flag_prima_cond == 0:

                            print('PRIMA CONDIZIONE')
                            mf_check_mesh_mid = MeshFunction('size_t', mesh_mid_ref, ndim, 0)
                            mf_check_mesh_mid.array()[test_mf_en] = 3
                            mf_check_mesh_mid.array()[test_mf_en_no_ref_mesh] = 1
                            if len(mark_en_search) > 0:
                                mf_check_mesh_mid.array()[test_mf_en_search_mesh] = 2

                            if len(mark_en_search) > 0:
                                add_cel_to_ref = np.concatenate(np.argwhere(mf_check_mesh_mid.array() == 2))
                            else:
                                add_cel_to_ref = np.concatenate(np.argwhere(mf_check_mesh_mid.array() == 1))

                            V_flag_ref = FunctionSpace(mesh_mid_ref, "DG", 0)
                            if flag_apply != 0:
                                f_apply_check = Function(V_flag_ref)
                                f_apply_check.rename('old_ref','old_ref')
                                LagrangeInterpolator.interpolate(f_apply_check, f_apply)
                                last_cell_refined = np.concatenate(np.argwhere(f_apply_check.vector() == 1)) #Vecchie celle effettivamente raffittite sulla nuova mesh

                            if len(mark_alpha_to_remove) == 0: #riduco vettore per ricerca elementi danneggiati
                                mark_alpha_to_check = np.concatenate(mark_alpha)
                            else:
                                V_alpha_remove = FunctionSpace(mesh1_ref, "DG", 0)
                                F_alpha_remove = Function(V_alpha_remove) #function definita sulla nuova mesh su cui andare a interpolare i vecchi elementi gia raffittiti
                                LagrangeInterpolator.interpolate(F_alpha_remove, f_apply) #interpolando marchio gli elementi raffittiti al passo/iterazione precedente
                                mark_alpha_remove = []
                                if len(np.argwhere(F_alpha_remove.vector() == 1))>0:
                                    mark_alpha_remove = np.concatenate(np.argwhere(F_alpha_remove.vector() == 1)) #elementi sulla mesh_ref "nuova" da rimuovere perche gia cercati
                                    mark_alpha_to_check = list(set(np.concatenate(mark_alpha)).difference(set(mark_alpha_remove)))
                                else:
                                    mark_alpha_to_check = np.concatenate(mark_alpha)

                            mf_alpha = MeshFunction('size_t', mesh1_ref, ndim, 0)
                            File(outputFolder + 'mesh_ref_cicles.pvd') << (mf_alpha, i_t*10000 + alternate_iter*10 + sub_cicles)

                            mf_alpha.array()[mark_alpha] = 1
                            mf_alpha.array()[mark_alpha_to_check] = 2

                            dom_ref_m3m = MeshFunction('bool', mesh_mid_ref, ndim, False)
                            cel = []
                            mark_alpha_to_recheck = []

                            for i in range(len(mark_alpha_to_check)):
                                G_a = Cell(mesh1_ref, mark_alpha_to_check[i]).midpoint()
                                for cell in add_cel_to_ref: 
                                    mark = Cell(mesh_mid_ref, cell).contains(G_a)
                                    if mark == True:
                                        cel.append([cell])
                                        mark_alpha_to_remove.append([mark_alpha_to_check[i]])
                                        break
                                    elif cell == add_cel_to_ref[len(add_cel_to_ref)-1] and mark == False:
                                        mark_alpha_to_recheck.append([mark_alpha_to_check[i]])

                            if len(mark_alpha_to_recheck) > 0:
                                mark_alpha_new = set(np.concatenate(mark_alpha)).difference(set(np.concatenate(mark_alpha_oldTs)))
                                mark_alpha_to_recheck = list(set(np.concatenate(mark_alpha_to_recheck)).difference(mark_alpha_new))
                                if len(mark_alpha_already_rechecked)>0:
                                    mark_alpha_to_recheck = list(set(mark_alpha_to_recheck).difference(set(mark_alpha_already_rechecked)))
                                for i in range(len(mark_alpha_to_recheck)):
                                    G_a = Cell(mesh1_ref, mark_alpha_to_recheck[i]).midpoint()
                                    cell_to_research = list(set(np.concatenate((np.argwhere(mf_check_mesh_mid.array() == 1), np.argwhere(mf_check_mesh_mid.array() == 1)), axis = None)).difference(set(mark_en_ref)))
                                    for celle in cell_to_research:
                                        mark = Cell(mesh_mid_ref, celle).contains(G_a)
                                        if mark == True:
                                            cel.append([celle])
                                            mark_alpha_already_rechecked.append(mark_alpha_to_recheck[i])
                                            break

                            if len(cel) > 0 or len(last_cell_refined) > 0:
                                cel_apply = []
                                if flag_apply == 0:
                                    cel_apply = np.concatenate(cel)
                                else:
                                    cel_apply_float = np.concatenate((cel, last_cell_refined), axis = None)
                                    cel_apply = list(map(int, cel_apply_float))
                                f_apply = Function(V_flag_ref)
                                f_apply.vector()[cel_apply] = 1
                                f_apply.rename('c_apply','c_apply')

                                dom_ref_m3m.array()[cel_apply] = True
                                mesh_ref_cicles = refine(mesh_mid_ref, dom_ref_m3m)
                                rmap = mesh_ref_cicles.data().array("parent_cell", ndim)
                                rmap_all.append(rmap)
                                mesh_mid_ref = mesh_ref_cicles

                            flag_search = 1
                            flag_apply = 1
                            flag_prima_cond = 1

                        else:
                            print('SECONDA CONDIZIONE')

                            V_flag_ref = FunctionSpace(mesh_mid_ref, "DG", 0)
                            f_apply_check = Function(V_flag_ref)
                            f_apply_check.rename('old_ref','old_ref')
                            LagrangeInterpolator.interpolate(f_apply_check, f_apply)
                            last_cell_refined = np.concatenate(np.argwhere(f_apply_check.vector() == 1))

                            dom_ref_m3m = MeshFunction('bool', mesh_mid_ref, ndim, False)
                            dom_ref_m3m.array()[last_cell_refined] = True
                            mesh_ref_cicles = refine(mesh_mid_ref, dom_ref_m3m)
                            rmap = mesh_ref_cicles.data().array("parent_cell", ndim)
                            rmap_all.append(rmap)
                            mesh_mid_ref = mesh_ref_cicles

                    mesh1_ref = mesh_mid_ref
                    if flag_load == 1:
                        flag_load_ref = 0
                        flag_already_ref = 1

                else:
                    mesh1_ref = mesh_mid_ref
                    if flag_load == 1:
                        flag_load_ref = 0           # ritorna zero per entrare in uguale
                        flag_already_ref = 1        # setto a 1 per non cambiare piu flag_load_ref a fine passo temporale

                mark_alpha_save_it = mark_alpha

#-----------------------------------------------------------------------------

                V_ur = FunctionSpace(mesh1_ref, V)
                ur = Function(V_ur)
                dur = TrialFunction(V_ur)
                vr = TestFunction(V_ur)
                ur.rename('ur','ur')

                boundaries_r = MeshFunction('size_t', mesh1_ref, ndim-1,0)
                domains_r = MeshFunction('size_t', mesh1_ref, ndim, 0)
                domains_r.set_all(0)
                core.mark(domains_r, 1)
                dxr = Measure('dx', domain=mesh1_ref, subdomain_data=domains_r)
                ds_fr = Measure('ds', domain=mesh1_ref, subdomain_data=boundaries_r)

                left.mark(boundaries_r, 1)
                top.mark(boundaries_r, 2)
                right.mark(boundaries_r, 3)
                bot.mark(boundaries_r, 4)

                bc_l = DirichletBC(V_ur.sub(0), Constant(0.), boundaries_r, 1)
                # bc_t = DirichletBC(V_ur.sub(1), Constant(0.), boundaries_r, 2)
                bc_r = DirichletBC(V_ur.sub(0), u_D, boundaries_r, 3)
                bc_b = DirichletBC(V_ur.sub(1), Constant(0.), boundaries_r, 4)
                #bcu_mid = DirichletBC(V_ur, Constant((0.,0.)), Node_M, method = 'pointwise') # Mid point
                # bcu_r = [bc_l, bc_t, bc_r, bc_b]
                bcu_r = [bc_l, bc_r, bc_b]

#------------------------------------------------------------------------------

                V_alpha_r = FunctionSpace(mesh1_ref, "P", 1)
                alpha_r = Function(V_alpha_r)
                alpha_r0 = Function(V_alpha_r)
                dalpha_r = TrialFunction(V_alpha_r)
                beta_r = TestFunction(V_alpha_r)
                alpha_r.rename('d_r','d_r')

                alpha_r_lb = Function(V_alpha_r)
                alpha_r_ub = Function(V_alpha_r)
                alpha_r_ub.vector()[:] = 1.

                if flag_tmp == 0:
                    print('flag_tmp 0 -> 1')
                    tmp = Function(V_alpha_r)
                    tmp_lb = Function(V_alpha_r)
                    tmp_lb.vector()[:] = 0.
                    flag_tmp += 1
                else:
                    LagrangeInterpolator.interpolate(alpha_r, tmp)
                alpha_r0.assign(alpha_r)
                LagrangeInterpolator.interpolate(alpha_r_lb, tmp_lb)

#------------------------------------------------------------------------------

                elastic_energy_r = 0.5 * inner(sigma(ur, alpha_r, E1, nu1, kres_1), eps(ur))*dxr(0)+\
                                   0.5 * inner(sigma(ur, alpha_r, E2, nu2, kres_1), eps(ur))*dxr(1)
                dissipated_energy_r = Gc1 / float(c_w) * (w(alpha_r) / ell1
                                      + ell1 * dot(grad(alpha_r), grad(alpha_r))) * dxr(0)+\
                                      Gc2 / float(c_w) * (w(alpha_r) / ell1
                                      + ell1 * dot(grad(alpha_r), grad(alpha_r))) * dxr(1)
                total_energy_r = elastic_energy_r + dissipated_energy_r

                # Derivative wrt u3
                E_ur = derivative(total_energy_r, ur, vr)
                E_dur = uf.replace(E_ur, {ur: dur})

                # First and second directional derivative wrt alpha
                E_dar = derivative(total_energy_r, alpha_r, beta_r)
                E_ddar = derivative(E_dar, alpha_r, dalpha_r)

#------------------------------------------------------------------------------

                problem_ur = LinearVariationalProblem(lhs(E_dur), rhs(E_dur), ur, bcu_r)
                solver_ur = LinearVariationalSolver(problem_ur)

                bcar = [DirichletBC(V_alpha_r, 0., boundaries_r, 3),
                        DirichletBC(V_alpha_r, 0., boundaries_r, 1)]
                problem_alpha_r = NonlinearVariationalProblem(E_dar, alpha_r, bcar, E_ddar)
                solver_alpha_r  = NonlinearVariationalSolver(problem_alpha_r)
                problem_alpha_r.set_bounds(alpha_r_lb, alpha_r_ub)

                # Solver for the alpha-problem
                pra = solver_alpha_r.parameters
                pra['nonlinear_solver'] = 'snes'
                pra["snes_solver"]["method"] = "vinewtonrsls"
                pra["snes_solver"]["maximum_iterations"] = 300
                pra["snes_solver"]["absolute_tolerance"] = 1E-7
                pra["snes_solver"]["relative_tolerance"] = 1E-7
                pra["snes_solver"]["linear_solver"] = "mumps"
                pra["snes_solver"]["line_search"] = "basic"
                pra["snes_solver"]["report"] = True

#------------------------------------------------------------------------------

            solver_ur.solve()
            solver_alpha_r.solve()
            File_dr_it << (alpha_r, i_t * 1000 + alternate_iter)

#------------------------------------------------------------------------------

            V_alpha_DG = FunctionSpace(mesh1_ref, 'DG', 0)
            alpha_r_DG = Function(V_alpha_DG)
            alpha_r_DG.rename('dr_DG','dr_DG')

            LagrangeInterpolator.interpolate(alpha_r_DG, alpha_r)
            mark_alpha = (np.argwhere(alpha_r_DG.vector() > alpha_soglia))
            flag_ma = len(mark_alpha)

#------------------------------------------------------------------------------

            # damage convergence alpha3
            err_alpha_r = (alpha_r.vector() - alpha_r0.vector()).norm('linf')
            norm = alpha_r0.vector().norm('l2')

            if (alpha_r0.vector().norm('l2') < 1.e-16):
                norm = 1.
            err_alpha_r_norm = (alpha_r.vector() - alpha_r0.vector()).norm('l2')/norm
            alpha_r_max = alpha_r.vector().max()

            if mpi_rank == 0:
                print("Iteration:  %2d" % (alternate_iter))
                print("Error1: %2.8g, alpha_r_max: %.8g" % (
                                  err_alpha_r, alpha_r_max,))
                print("Error2: %2.8g, alpha3_max: %.8g" % (
                                  err_alpha_r_norm, alpha_r_max,))

            if err_alpha_r < alternate_conv_tol:
                alternate_convergence = True

            # save old vectors
            tmp = alpha_r
            mark_en_save = mark_en
            mark_alpha_save = mark_alpha

            mark_search_save = mark_en_search

            LagrangeInterpolator.interpolate(alpha, alpha_r)

            if flag_final_ref == 1:
                flag_final_ref = 0

            if flag_ref == 1:
                flag_ref = 0

            domains.set_all(0)

#------------------------------------------------------------------------------

        elif d_en.min() >= 0:
            print('d_en.min >= 0')

            problem_alpha = NonlinearVariationalProblem(E_da, alpha, bca, E_dda)
            solver_alpha  = NonlinearVariationalSolver(problem_alpha)
            problem_alpha.set_bounds(alpha_lb, alpha_ub)

            # Solver for the alpha-problem
            pra = solver_alpha.parameters
            pra['nonlinear_solver'] = 'snes'
            pra["snes_solver"]["method"] = "vinewtonrsls"
            pra["snes_solver"]["maximum_iterations"] = 300
            pra["snes_solver"]["absolute_tolerance"] = 1E-7
            pra["snes_solver"]["relative_tolerance"] = 1E-7
            pra["snes_solver"]["linear_solver"] = "mumps"
            pra["snes_solver"]["line_search"] = "basic"
            pra["snes_solver"]["report"] = True

            solver_alpha.solve()

            # damage convergence alpha3
            err_alpha = (alpha.vector() - alpha_0.vector()).norm('linf')
            alpha_max = alpha.vector().max()

            if mpi_rank == 0:
                print("Iteration:  %2d" % (alternate_iter))
                print("Error1: %2.8g, alpha1_max: %.8g" % (
                                  err_alpha, alpha_max,))

            if err_alpha < alternate_conv_tol:
                alternate_convergence = True

            alpha_0.assign(alpha)

#------------------------------------------------------------------------------

    if d_en.min() < 0:
        tmp_lb = alpha_r
        print('update d_en_back')

    if d_en.min() < 0 and alpha_r.vector().max() > 0.99:
        flag_ref = 1

#------------------------------------------------------------------------------

# ---Update vectors
    mark_en_oooold = mark_en_ooold
    mark_en_ooold = mark_en_oold
    mark_en_oold = mark_en_old
    mark_en_old = mark_en

    if flag_refine > 0:
        mark_en_no_ref_old = mark_en_no_ref

    if flag_ma > 0:
        flag_refine = 1

    alpha_lb.vector()[:]=alpha.vector()

    flag_mesh3_ref = 0
    flag_search = 0

    postprocessing()

    if flag_ma > 0 and max(alpha_r.vector())> 0.999:
        flag_load = 1                   # blocca il carico
        if flag_already_ref == 0:        # check che abbia gia fatto il ref post lock carico
            flag_load_ref = 1           # fa entrare nel refine post blocco carico

    if mpi_rank == 0:
        print("\nEnd of timestep %d with load %g" % (i_t, t))