from fenics import *
import mshr as mr
import shutil
from mpi4py import MPI
import numpy as np
from scipy import spatial
import ufl as uf
import sys, os, sympy, shutil, math

outputFolder = "AGu_AGal/"

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

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

# --- Useful functions
def dom_param_listing(dom_list, E_list, nu_list, ell_list, Gc_list, ww_list):
    """Associate the elasticity/damage parameter with the corresponding domain"""
    for bl in range(len(dom_list)):
        exec(f'E{bl} = E_list[bl]')
        exec(f'nu{bl} = nu_list[bl]')
        exec(f'ell{bl} = ell_list[bl]')
        exec(f'Gc{bl} = Gc_list[bl]')
        exec(f'ww{bl} = ww_list[bl]')

def doms(dom_list, noDmg_dom, domain):
    """Mark the internal domains with and without damage"""
    j = 0
    for dm in dom_list:
        dm.mark(domain, j)
        j += 1

    i = len(dom_list)
    for bl in noDmg_dom:
        bl.mark(domain, i)
        i += 1

def bound(bound_list, free_bound, noDmg, subdomain):
    """Marks the facets: 1) where no dmg, 2) if free boundaries are required (eg: pre-existing cracks within the solid)
                             3) subdomains which are requested as input"""
    for bl in noDmg:
        bl.mark(subdomain, 1000)
    for bl in free_bound:
        bl.mark(subdomain, len(bound_list)+1)
    i = 1
    for bl in bound_list:
        bl.mark(subdomain, i)
        i += 1

def boundary_conditions(V_u, V_alpha, boundaries, bc_u, bc_a, noDmg):
    """Apply the displacement and damage dirichlet BC set by the user on the coarse mesh"""
    if len(noDmg) > 0:
        bc_a.append([0, noDmg, 1000])
    bcu = []
    bca = []
    for bc in bc_u:
        if (type(bc[1]) == int or type(bc[1]) == float) and type(bc[-1]) == int:        # costante applicata ad un bordo
            bcu.append(DirichletBC(V_u.sub(bc[0]), Constant(bc[1]), boundaries, bc[-1]))
        elif (type(bc[1]) != int or type(bc[1]) != float) and type(bc[-1]) == int:      # espressione applicata ad un bordo
            bcu.append(DirichletBC(V_u.sub(bc[0]), bc[1], boundaries, bc[-1]))
        elif (type(bc[1]) == int or type(bc[1]) == float):                              # costante applicata ad un nodo
            bcu.append(DirichletBC(V_u.sub(bc[0]), Constant(bc[1]), bc[2], method = bc[-1]))
        else:                                                                           # espressione aplicata ad un nodo
            bcu.append(DirichletBC(V_u.sub(bc[0]), bc[1], bc[2], method = bc[-1]))
    for bc in bc_a:
        if (type(bc[0]) == int or type(bc[0]) == float) and type(bc[-1]) == int:        # costante applicata ad un bordo
            bca.append(DirichletBC(V_alpha, Constant(bc[0]), boundaries, bc[-1]))
        elif (type(bc[0]) != int or type(bc[0]) != float) and type(bc[-1]) == int:      # espressione applicata ad un bordo
            bca.append(DirichletBC(V_alpha, bc[0], boundaries, bc[-1]))
        elif (type(bc[0]) == int or type(bc[0]) == float):                              # costante applicata ad un nodo
            bca.append(DirichletBC(V_alpha, Constant(bc[0]), bc[1], method = bc[-1]))
        else:                                                                           # espressione aplicata ad un nodo
            bca.append(DirichletBC(V_alpha, bc[0], bc[1], method = bc[-1]))
    return bcu, bca

def boundary_conditions_ref(V_ur, V_alpha_r, boundaries_r, bc_u, bc_a, noDmg):
    """Apply the displacement and damage dirichlet BC set by the user on the refined mesh"""
    if len(noDmg) > 0:
        bc_a.append([0, noDmg, 1000])
    bcu = []
    bca = []
    for bc in bc_u:
        if (type(bc[1]) == int or type(bc[1]) == float) and type(bc[-1]) == int:        # costante applicata ad un bordo
            bcu.append(DirichletBC(V_ur.sub(bc[0]), Constant(bc[1]), boundaries_r, bc[-1]))
        elif (type(bc[1]) != int or type(bc[1]) != float) and type(bc[-1]) == int:      # espressione applicata ad un bordo
            bcu.append(DirichletBC(V_ur.sub(bc[0]), bc[1], boundaries_r, bc[-1]))
        elif (type(bc[1]) == int or type(bc[1]) == float):                              # costante applicata ad un nodo
            bcu.append(DirichletBC(V_ur.sub(bc[0]), Constant(bc[1]), bc[2], method = bc[-1]))
        else:                                                                           # espressione aplicata ad un nodo
            bcu.append(DirichletBC(V_ur.sub(bc[0]), bc[1], bc[2], method = bc[-1]))
    for bc in bc_a:
        if (type(bc[0]) == int or type(bc[0]) == float) and type(bc[-1]) == int:        # costante applicata ad un bordo
            bca.append(DirichletBC(V_alpha_r, Constant(bc[0]), boundaries_r, bc[-1]))
        elif (type(bc[0]) != int or type(bc[0]) != float) and type(bc[-1]) == int:      # espressione applicata ad un bordo
            bca.append(DirichletBC(V_alpha_r, bc[0], boundaries_r, bc[-1]))
        elif (type(bc[0]) == int or type(bc[0]) == float):                              # costante applicata ad un nodo
            bca.append(DirichletBC(V_alpha_r, Constant(bc[0]), bc[1], method = bc[-1]))
        else:                                                                           # espressione aplicata ad un nodo
            bca.append(DirichletBC(V_alpha_r, bc[0], bc[1], method = bc[-1]))
    return bcu, bca

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

# --- Elasticity definitions
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

def sigma_0(u, E, nu, ndim):
    mu = E / (2. * (1. + nu))
    lamb = E * nu / (1. - nu ** 2)
    return 2. * mu * (eps(u)) + lamb * tr(eps(u)) * Identity(ndim+1)

def sigma(u, alpha, E, nu, kres, ndim):
    return (a(alpha, kres)) * sigma_0(u, E, nu, ndim)

z = sympy.Symbol("z")
c_w = 4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1))

# --- Refinement function solver
def AGu_AGal(mesh , E, nu, ell, Gc, E_noDmg, nu_noDmg, bound_list, dom_list, noDmg_doms, bc_u, bc_a, u_D, loads, ref_param):
    kres_1 = Constant(1e-10)

    # --- cell_size, tree
    print('Cell size')
    cell_size = Cell(mesh,0).h()
    ndim = mesh.topology().dim()
    G = []
    for ii in cells(mesh):
        G.append([ii.midpoint().x(), ii.midpoint().y()])
    tree = spatial.cKDTree(G)

    # --- Modifica numero cicli di raffittimento
    print('Refinement Parameter')
    beta_1, beta_2, beta_3, alpha_soglia = 1, 1.1, 1.1, ref_param[0]
    coeff_mark_en = beta_3 * cell_size
    alpha_s = alpha_soglia

    ref_in = ref_param[1]
    ref_add = ref_param[2]
    ref_active = ref_in-ref_add

    # --- Function Spaces
    V = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    V_u = FunctionSpace(mesh, V)
    u = Function(V_u)
    du = TrialFunction(V_u)
    v = TestFunction(V_u)
    u.rename('u', 'u')
    
    V_alpha = FunctionSpace(mesh, "CG", 1)
    alpha = Function(V_alpha)
    dalpha = TrialFunction(V_alpha)
    beta = TestFunction(V_alpha)
    alpha.rename('damage', 'damage')
    alpha_0 = Function(V_alpha)
    alpha_lb = interpolate(Constant("0."), V_alpha)
    alpha_ub = interpolate(Constant("1."), V_alpha)
    
    V_Source = FunctionSpace(mesh, 'DG', 0)
    V_flag = FunctionSpace(mesh, "DG", 0)
    alpha_DG = Function(V_flag)

    # ---Displacement BC
    print("Boundary Conditions")
    free_bound = []
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    bound(bound_list, free_bound, noDmg_doms, boundaries)
    bcu, bca = boundary_conditions(V_u, V_alpha, boundaries, bc_u, bc_a, noDmg_doms)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    doms(dom_list, noDmg_doms, domains)
    dx = Measure('dx', domain=mesh, subdomain_data=domains)

    # ---UFL Forms
    print("UFL forms")
    elastic_energy = 0*dx
    dissipated_energy = 0*dx
    if len(dom_list) > 0:
        for i in range(len(dom_list)):
            elastic_energy += 0.5 * inner(sigma(u, alpha, E[i], nu[i], kres_1, ndim), eps(u))*dx(i)(metadata={'quadrature_degree': 5})
            dissipated_energy += Gc[i] / float(c_w) * (w(alpha) / ell[i] + ell[i] * dot(grad(alpha), grad(alpha)))*dx(i)
    else:
        elastic_energy = 0.5 * inner(sigma(u, alpha, E[0], nu[0], kres_1, ndim), eps(u))*dx(0)(metadata={'quadrature_degree': 5})
        dissipated_energy = Gc[0] / float(c_w) * (w(alpha) / ell[0] + ell[0] * dot(grad(alpha), grad(alpha))) * dx(0)

    if len(noDmg_doms) > 0:
        for i in range(len(dom_list), len(dom_list) + len(noDmg_doms)):
            elastic_energy += 0.5 * inner(sigma_0(u, E_noDmg[i-len(dom_list)], nu_noDmg[i-len(dom_list)], ndim), eps(u))*dx(i)(metadata={'quadrature_degree': 5})
            dissipated_energy += Constant(0.) * dot(grad(alpha), grad(alpha)) * dx(i)

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

    # --- Displacement savefile
    File_u = File(outputFolder + "/u.pvd")
    File_ur = File(outputFolder + "/u_r.pvd")

    # --- Damage savefile
    File_d = File(outputFolder + "/d.pvd")
    File_dr = File(outputFolder + "/dr.pvd")
    File_dr_it = File(outputFolder + "/dr_it.pvd")

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

    mark_en = []
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

    mark_add = []

    mark_alpha_for_ref = []
    ma_for_ref = []
    ma_oldTs_base = []

    alternate_conv_tol = 2.e-5
    G_alpha = []

    save_err = 0
    save_err_old = 0

    maxiter = 1000

    for (i_t, t) in enumerate(loads):
        u_D.t = t

        if t == loads[-1]:
            flag_final_ref = 1

        if list(mark_alpha) != list(mark_alpha_oldTs):
            print('Mark_alpha -> New elements for alpha_s refinement')
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
                doms(dom_list, noDmg_doms, domains)
                V_Source = FunctionSpace(mesh, "DG", 0)
                d_en = Function(V_Source)
                for i in range(len(dom_list)):
                    print(i)
                    En_rif = 3/8 * Gc[i]/ ell[i]
                    En = inner(sigma_0(u, E[i], nu[i], ndim ), eps(u))
                    En_projected = project(En, V_Source)
                    source = project(beta_1 * En_rif - En_projected, V_Source)
                    dom_mask = np.concatenate(np.argwhere(domains.array() == i))
                    d_en.vector()[dom_mask] = source.vector()[dom_mask]
                if len(noDmg_doms)>0:
                    dom_noDmg_mask = np.concatenate(np.argwhere(domains.array() >= len(dom_list)))
                    d_en.vector()[dom_noDmg_mask] = 0
            else :
                V_Source = FunctionSpace(mesh1_ref, "DG", 0)
                d_en_r = Function(V_Source)
                for i in range(len(dom_list)):
                    En_rif = 3/8 * Gc[i]/ ell[i]
                    En_r = inner(sigma_0(ur, E[i], nu[i], ndim ), eps (ur))
                    En_r_projected = project(En_r, V_Source)
                    source_r = project(beta_1 * En_rif - En_projected, V_Source)
                    dom_mask = np.concatenate(np.argwhere(domains_r.array() == i))
                    d_en_r.vector()[dom_mask] = source_r.vector()[dom_mask]
                if len(noDmg_doms)>0:
                    dom_noDmg_mask = np.concatenate(np.argwhere(domains_r.array() >= len(dom_list)))
                    d_en_r.vector()[dom_noDmg_mask] = 0

    #------------------------------------------------------------------------------

            if d_en.vector().min() < 0:
                print('d_en.min < 0')
                if True:
                    if flag_ma > 0:
                        LagrangeInterpolator.interpolate(u, ur)
                    doms(dom_list, noDmg_doms, domains)
                    V_Source = FunctionSpace(mesh, "DG", 0)
                    d_en = Function(V_Source)
                    for i in range(len(dom_list)):
                        print(i)
                        En_rif = 3/8 * Gc[i]/ ell[i]
                        En = inner(sigma_0(u, E[i], nu[i], ndim ), eps(u))
                        En_projected = project(En, V_Source)
                        source = project(beta_1 * En_rif - En_projected, V_Source)
                        dom_mask = np.concatenate(np.argwhere(domains.array () == i))
                        d_en.vector()[dom_mask] = source.vector()[dom_mask]
                    if len(noDmg_doms)>0:
                        dom_noDmg_mask = np.concatenate(np.argwhere(domains.array() >= len(dom_list)))
                        d_en.vector()[dom_noDmg_mask] = 0
                    mark_en = np.concatenate(np.argwhere(d_en.vector()[:] < 0))
                else:
                    if flag_ma == 0:
                        mark_en = np.concatenate(np.argwhere(d_en.vector()[:] < 0))
                    else :
                        mark_en_r = np.concatenate(np.argwhere(d_en_r.vector()[:] < 0))
                        for ll in range(len(rmap_all)):
                            rmap_array = np.array(rmap_all[len(rmap_all) - 1 - ll])
                            mark_en_coarse = rmap_array[mark_en_r]
                            mark_en_r = mark_en_coarse
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
                mark_en_ref = list(set(mark_en) - set(mark_en_no_ref))

                if list(mark_en) == list(mark_en_save) and flag_final_ref == 0:
                    print('Same Mesh')

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
                    print('New elements to be refined')
                    mark_add_prev_iter = mark_add
                    mark_add = []
                    for el_id in mark_en:
                        p_x = Cell(mesh, el_id).midpoint().x()
                        p_y = Cell(mesh, el_id).midpoint().y()
                        pp = [p_x, p_y]
                        mark_add.append(tree.query_ball_point(pp, coeff_mark_en, np.float('inf')))
                    mark_add = list(set(np.concatenate(mark_add)))

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

                                print('First condition -> full loop for alpha_s refinement')
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
                                print('Second condition -> only interpolation')

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

                        mesh1_ref = mesh_ref_cicles
                        if flag_load == 1:
                            flag_load_ref = 0
                            flag_already_ref = 1

                    else:
                        mesh1_ref = mesh_mid_ref
                        if flag_load == 1:
                            flag_load_ref = 0
                            flag_already_ref = 1

                    mark_alpha_save_it = mark_alpha

    #-----------------------------------------------------------------------------

                    V_ur = FunctionSpace(mesh1_ref, V)
                    ur = Function(V_ur)
                    dur = TrialFunction(V_ur)
                    vr = TestFunction(V_ur)
                    ur.rename('ur','ur')

                    boundaries_r = MeshFunction('size_t', mesh1_ref, ndim-1,0)
                    domains_r = MeshFunction('size_t', mesh1_ref, ndim, 0)
                    doms(dom_list, noDmg_doms, domains_r)
                    dxr = Measure('dx', domain=mesh1_ref, subdomain_data=domains_r)

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

    # -----------------------------------------------------------------------------

                    bound(bound_list, free_bound, noDmg_doms, boundaries_r)
                    bcu_r, bcar = boundary_conditions_ref(V_ur, V_alpha_r, boundaries_r, bc_u, bc_a, noDmg_doms)

    #------------------------------------------------------------------------------

                    elastic_energy_r = 0*dxr
                    dissipated_energy_r = 0*dxr

                    if len(dom_list) > 0:
                        for i in range(len(dom_list)):
                            elastic_energy_r += 0.5 * inner(sigma(ur, alpha_r, E[i], nu[i], kres_1, ndim), eps(ur))*dxr(i)(metadata={'quadrature_degree': 5})
                            dissipated_energy_r += Gc[i] / float(c_w) * (w(alpha_r) / ell[i] + ell[i] * dot(grad(alpha_r), grad(alpha_r)))*dxr(i)
                    else:
                        elastic_energy_r = 0.5 * inner(sigma(ur, alpha_r, E[0], nu[0], kres_1, 2), eps(ur))*dxr(0)(metadata={'quadrature_degree': 5})
                        dissipated_energy_r = Gc[0] / float(c_w) * (w(alpha_r) / ell[0] + ell[0] * dot(grad(alpha_r), grad(alpha_r))) * dxr(0)

                    if len(noDmg_doms) > 0:
                        for i in range(len(dom_list), len(dom_list) + len(noDmg_doms)):
                            elastic_energy_r += 0.5 * inner(sigma_0(ur, E_noDmg[i-len(dom_list)], nu_noDmg[i-len(dom_list)], ndim), eps(ur))*dxr(i)(metadata={'quadrature_degree': 5})
                            dissipated_energy_r += Constant(0.) * dot(grad(alpha_r), grad(alpha_r)) * dxr(i)
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
                File_dr_it << (alpha_r, i_t*1000 + alternate_iter)

    #------------------------------------------------------------------------------

                if list(mark_en) != list(mark_en_save):
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

            elif d_en.vector().min() >= 0:
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

        if d_en.vector().min() < 0:
            tmp_lb = alpha_r
            print('update d_en_back')

        if d_en.vector().min() < 0 and alpha_r.vector().max() > 0.99:
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

        if (flag_ma > 0):
            File_ur << (ur, t)
            File_dr << (alpha_r, i_t * 1000 + alternate_iter)
            elastic_energy_value = assemble(elastic_energy_r)
            surface_energy_value = assemble(dissipated_energy_r)
        else:
            File_u << (u, t)
            File_d << (alpha, t)
            elastic_energy_value = assemble(elastic_energy)
            surface_energy_value = assemble(dissipated_energy)
        energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value + surface_energy_value])
        np.savetxt(outputFolder + '/energies.txt', energies)

        if flag_ma > 0 and max(alpha_r.vector())> 0.999:
            flag_load = 1                   
            if flag_already_ref == 0:
                flag_load_ref = 1

        if mpi_rank == 0:
            print("\nEnd of timestep %d with load %g" % (i_t, t))