import casadi as cd
from do_mpc.model import Model
from RL_Tools.tools.RL_MPC import Actor_MPC as MPC

def build_benchmark_MPC_model():
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = Model(model_type)

    # Certain parameters
    R           = 8.314    			    # gas constant [J/K*mol]
    A_tank      = 65.0       			# heat exchanger surface jacket [m2]
    m_M_KW      = 5000.0                # mass of coolant in jacket [kg]
    m_AWT_KW    = 1000.0                # mass of coolant in external heat exchanger [kg]
    m_AWT       = 200.0                 # mass of product in external heat exchanger [kg]
    m_S         = 39000.0               # mass of reactor steel [kg]
    c_pW        = 4.2                   # specific heat capacity of the coolant [kJ/kg*K]
    c_pS        = 0.47                  # specific heat capacity of steel (vessel) [kJ/kg*K]
    c_pF        = 3.0                   # specific heat capacity of feed [kJ/kg*K]
    c_pR        = 5.0                   # specific heat capacity of reactor contents (mixture) [kJ/kg*K]

    # Certain parameters (maybe external disturbances in the future)
    T_F         = 25 + 273.15           # feed temperature [K]
    w_WF        = 0.333                  # Mass fraction water in feed [-]
    w_AF        = 0.667                  # Mass fraction of Monomer A in feed [-]

    fm_M_KW     = 300000.0              # coolant flow in jacket [kg/h];
    fm_AWT_KW   = 100000.0              # coolant flow in external heat exchanger [kg/h]
    fm_AWT      = 20000.0               # mixture flow in external heat exchanger [kg/h]

    # Potentiall uncertain parameters (currently certain)
    E_a         = 8500.0     			# Specific activation energy [kJ/kg]
    delH_R      = 950.0*1.00      		# Specific heat of reaction [kJ/kg]

    k_0         = 7.0*1.00      	    # Specific reaction rate [Check units]
    k_U2        = 32.0     	            # Reaction parameter 1 [Check units]
    k_U1        = 4.0      	            # Reaction parameter 2 [Check units]

    k_WS        = 17280.0               # heat transfer coefficient water-steel [kJ/h*m2*K]
    k_AS        = 3600.0                # heat transfer coefficient monomer-steel [kJ/h*m2*K]
    k_PS        = 360.0                 # heat transfer coefficient product-steel [kJ/h*m2*K]

    alfa        = 5*20e4*3.6            # Experimental coefficient [1/h]
    p_1         = 1.0                   # ???

    # States struct (optimization variables):
    m_W =         model.set_variable('_x', 'm_W')   # Mass of water inside the reactor [kg]
    m_A =         model.set_variable('_x', 'm_A')   # Mass of monomer inside the reactor [kg]
    m_P =         model.set_variable('_x', 'm_P')   # Mass of product inside the reactor [kg]

    T_R =         model.set_variable('_x', 'T_R')       # Temperature of the mixture inside the reactor [K]
    T_S =         model.set_variable('_x', 'T_S')       # Temperature of vessel (steel) [K]
    Tout_M =      model.set_variable('_x', 'Tout_M')    # Temperature of coolant in the jacket [K]
    T_EK =        model.set_variable('_x', 'T_EK')      # Temperature of mixture in the external heat exchanger [K]
    Tout_AWT =    model.set_variable('_x', 'Tout_AWT')  # Temperature of coolant in the external heat exchanger [K]

    accum_monom = model.set_variable('_x', 'accum_monom')   # Mass of accumulated monomer [kg]
    T_adiab =     model.set_variable('_x', 'T_adiab')       # Adiabatic temperature [K]

    # Input struct (optimization variables):
    m_dot_f = model.set_variable('_u', 'm_dot_f')   # Mass feed flowrate [kg/h]
    T_in_M =  model.set_variable('_u', 'T_in_M')    # Coolant temperature at the inlet of the jacket [K]
    T_in_EK = model.set_variable('_u', 'T_in_EK')   # Coolant temperature at the inlet of the external heat exchanger [K]


    # algebraic equations
    U_m    = m_P / (m_A + m_P)
    m_ges  = m_W + m_A + m_P
    k_R1   = k_0 * cd.exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_R2   = k_0 * cd.exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)

    # Differential equations
    dot_m_W = m_dot_f * w_WF
    model.set_rhs('m_W', dot_m_W)
    dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
    model.set_rhs('m_A', dot_m_A)
    dot_m_P = (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
    model.set_rhs('m_P', dot_m_P)

    dot_T_R = 1./(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))))
    model.set_rhs('T_R', dot_T_R)
    model.set_rhs('T_S', 1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M))))
    model.set_rhs('Tout_M', 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M))))
    model.set_rhs('T_EK', 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R)))
    model.set_rhs('Tout_AWT', 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK))))
    model.set_rhs('accum_monom', m_dot_f)
    model.set_rhs('T_adiab', delH_R/(m_ges*c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+dot_T_R)

    # Build the model
    model.setup()

    return model

def build_benchmark_MPC(model):
    mpc = MPC(model)

    mpc.settings.n_horizon = 30  # Currently this means that the controller looks 20 steps/minutes ahead
    mpc.settings.open_loop = False
    mpc.settings.t_step = 30.0/3600.0 # 60 seconds in hours
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.store_full_solution = True
    mpc.settings.collocation_ni = 2    

    x = mpc.model.x

    mterm = - model.x['m_P']
    lterm = - model.x['m_P']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002)
    # mpc.set_rterm(m_dot_f=0., T_in_M=0., T_in_EK=0.) # NOTE: This can be done because we have the max change rate in the bottom

    temp_range = 2.0
    T_ref = 273.15 + 90.0

    # mpc.bounds['lower','_x','m_W'] = 0.0
    # mpc.bounds['lower','_x','m_A'] = 0.0
    # mpc.bounds['lower','_x','m_P'] = 0.0

    # mpc.bounds['lower','_x','T_R'] = T_ref - temp_range
    # mpc.bounds['lower','_x','T_S'] = 273.15 + 20.0
    # mpc.bounds['lower','_x','Tout_M'] = 273.15 + 20.0
    # mpc.bounds['lower','_x','T_EK'] = 273.15 + 15.0
    # mpc.bounds['lower','_x','Tout_AWT'] = 273.15 + 15.0
    # mpc.bounds['lower','_x','accum_monom'] = 0.0
    # # mpc.bounds['lower','_x','T_adiab'] = 273.15 + 0.0
    

    # mpc.bounds['upper','_x','T_S'] = 273.15 + 100.0
    # mpc.bounds['upper','_x','Tout_M'] = 273.15 + 100.0
    # mpc.bounds['upper','_x','T_EK'] = 273.15 + 100.0
    # mpc.bounds['upper','_x','Tout_AWT'] = 273.15 + 100.0
    # mpc.bounds['upper','_x','accum_monom'] = 30e3
    # mpc.bounds['upper','_x','T_adiab'] = 273.15 + 109.0

    # mpc.bounds['lower','_u','m_dot_f'] = 0.0
    # mpc.bounds['lower','_u','T_in_M'] = 273.15 + 60.0
    # mpc.bounds['lower','_u','T_in_EK'] = 273.15 + 60.0

    # mpc.bounds['upper','_u','m_dot_f'] = 30e3
    # mpc.bounds['upper','_u','T_in_M'] = 273.15 + 100.0
    # mpc.bounds['upper','_u','T_in_EK'] = 273.15 + 100.0

    mpc.bounds['lower','_u','m_dot_f'] = 0.0
    mpc.bounds['lower','_u','T_in_M'] = 273.15 + 60.0
    mpc.bounds['lower','_u','T_in_EK'] = 273.15 + 60.0

    mpc.bounds['upper','_u','m_dot_f'] = 3.0e4
    mpc.bounds['upper','_u','T_in_M'] = 273.15 + 100.0
    mpc.bounds['upper','_u','T_in_EK'] = 273.15 + 100.0

    
    weight_x = 1e4

    # reg_type = "l2"
    # max_viol_m = 1.0
    # max_viol_T = 1.0

    reg_type = "l1"
    max_viol_m = cd.inf
    max_viol_T = cd.inf
    # max_viol_m = 100
    # max_viol_T = 10


    mpc.bounds['lower','_x','m_W'] = 0.0
    mpc.bounds['lower','_x','m_A'] = 0.0
    mpc.bounds['lower','_x','m_P'] = 0.0
    mpc.bounds['lower','_x','accum_monom'] = 0.0

    mpc.bounds['upper','_x','m_W'] = 30.0e3
    mpc.bounds['upper','_x','m_A'] = 30.0e3
    mpc.bounds['upper','_x','m_P'] = 30.0e3
    mpc.bounds['upper','_x','accum_monom'] = 30.0e3

    mpc.bounds['lower','_x','T_R'] = T_ref - temp_range
    mpc.bounds['lower','_x','T_S'] = 273.15 + 60.0
    mpc.bounds['lower','_x','Tout_M'] = 273.15 + 60.0
    mpc.bounds['lower','_x','T_EK'] = 273.15 + 60.0
    mpc.bounds['lower','_x','Tout_AWT'] = 273.15 + 60.0
    mpc.bounds["lower", "_x", "T_adiab"] = 273.15

    # mpc.bounds['upper','_x','T_R'] = T_ref + temp_range
    mpc.bounds['upper','_x','T_S'] = 273.15 + 100.0
    mpc.bounds['upper','_x','Tout_M'] = 273.15 + 100.0
    mpc.bounds['upper','_x','T_EK'] = 273.15 + 100.0
    mpc.bounds['upper','_x','Tout_AWT'] = 273.15 + 100.0
    mpc.bounds["upper", "_x", "T_adiab"] = 273.15 + 109.0



    #mpc.set_nl_cons(expr_name = "m_W_lower_limit", expr = 0. - x['m_W'], ub = 0., soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m, regularization_type = reg_type)
    #mpc.set_nl_cons(expr_name = "m_A_lower_limit", expr = 0. - x['m_A'], ub = 0., soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m,regularization_type = reg_type)
    #mpc.set_nl_cons(expr_name = "m_P_lower_limit", expr = 0. - x['m_P'], ub = 0., soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m,regularization_type = reg_type)

    # mpc.set_nl_cons(expr_name = "m_W_upper_limit", expr = x['m_W'] - 30.0e3, ub = 0, soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m, regularization_type = reg_type) # NOTE: Actually they are unconstrained
    # mpc.set_nl_cons(expr_name = "m_A_upper_limit", expr = x['m_A'] - 30.0e3, ub = 0, soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m, regularization_type = reg_type) # NOTE: Actually they are unconstrained
    # mpc.set_nl_cons(expr_name = "m_P_upper_limit", expr = x['m_P'] - 30.0e3, ub = 0, soft_constraint = True, penalty_term_cons = weight_x, maximum_violation = max_viol_m, regularization_type = reg_type) # NOTE: Actually they are unconstrained

    # mpc.set_nl_cons(expr_name = "T_R_lower_limit", expr = (T_ref - temp_range) - x['T_R'], ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "T_S_lower_limit", expr = 273.15 + 20.0 - x['T_S'], ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "Tout_M_lower_limit", expr = 273.15 + 20.0 - x['Tout_M'], ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "T_EK_lower_limit", expr = 273.15 + 15.0 - x['T_EK'], ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "Tout_AWT_lower_limit", expr = 273.15 + 15.0 - x['Tout_AWT'], ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "accum_monom_lower_limit", expr = 0.0 - x['accum_monom'], ub = 0., soft_constraint = True, penalty_term_cons = weight_x,  regularization_type = reg_type)
    mpc.set_nl_cons(expr_name = "T_adiab_lower_limit", expr = x['T_R'] - x['T_adiab'], ub = 0, soft_constraint = False)

    mpc.set_nl_cons(expr_name = "T_R_upper_limit", expr = x['T_R'] -  (T_ref + temp_range), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "T_S_upper_limit", expr = x['T_S'] - (273.15 + 100.0), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "Tout_M_upper_limit", expr = x['Tout_M'] - (273.15 + 100.0), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "T_EK_upper_limit", expr = x['T_EK'] - (273.15 + 100.0), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "Tout_AWT_upper_limit", expr = x['Tout_AWT'] - (273.15 + 100.0), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "accum_monom_upper_limit", expr = x['accum_monom'] - 30.0e3, ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)
    # mpc.set_nl_cons(expr_name = "T_adiab_upper_limit", expr = x['T_adiab'] - (273.15 + 109.0), ub = 0, soft_constraint = True, penalty_term_cons = weight_x, regularization_type = reg_type)


    # Scaling
    # mpc.scaling['_x','m_W'] = 10
    # mpc.scaling['_x','m_A'] = 10
    # mpc.scaling['_x','m_P'] = 10
    # mpc.scaling['_x','accum_monom'] = 10

    # mpc.scaling['_u','m_dot_f'] = 100

    mpc.scaling['_x','m_W'] = 10e3
    mpc.scaling['_x','m_A'] = 10e3
    mpc.scaling['_x','m_P'] = 10e3
    mpc.scaling['_x','accum_monom'] = 10e3
    
    mpc.scaling["_x", "T_R"] = 100.0
    mpc.scaling["_x", "T_S"] = 100.0
    mpc.scaling["_x", "Tout_M"] = 100.0
    mpc.scaling["_x", "T_EK"] = 100.0
    mpc.scaling["_x", "Tout_AWT"] = 100.0
    mpc.scaling["_x", "T_adiab"] = 100.0


    mpc.scaling['_u','m_dot_f'] = 10e3
    mpc.scaling['_u','T_in_M'] = 100.0
    mpc.scaling['_u','T_in_EK'] = 100.0

    
    # Check if robust multi-stage is active
    # if mpc.settings.n_robust == 0:
    #     # Sot-constraint for the reactor upper bound temperature
    #     mpc.set_nl_cons('T_R_UB', model.x['T_R'], ub = T_ref + temp_range, soft_constraint = True, penalty_term_cons = weight_x)
    # else:
    #     mpc.bounds['upper','_x','T_R'] = T_ref + temp_range

    mpc.settings.nlpsol_opts.update({"ipopt.fixed_variable_treatment": "make_parameter",})
    mpc.settings.nlpsol_opts.update({"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False })

    # mpc.setup()

    mpc.prepare_nlp()

    # du_dt_lb = np.array([-1500 * 60, -20 * 60, -20 * 60])
    # du_dt_lb = du_dt_lb * mpc.settings.t_step
    # du_dt_lb = du_dt_lb / mpc.opt_x_scaling["_u", 0, 0]

    # du_dt_ub = np.array([1500 * 60, 20 * 60, 20 * 60])
    # du_dt_ub = du_dt_ub * mpc.settings.t_step
    # du_dt_ub = du_dt_ub / mpc.opt_x_scaling["_u", 0, 0]

    # mpc._nlp_cons.append(mpc.opt_x["_u", 0, 0] - mpc.opt_p["_u_prev"]/mpc.opt_x_scaling["_u", 0, 0])
    # mpc._nlp_cons_lb.append(du_dt_lb)
    # mpc._nlp_cons_ub.append(du_dt_ub)

    # for i in range(mpc.settings.n_horizon - 1):
    #     cons = mpc.opt_x["_u", i + 1, 0] - mpc.opt_x["_u", i, 0]

    #     mpc._nlp_cons.append(cons)
    #     mpc._nlp_cons_lb.append(du_dt_lb)
    #     mpc._nlp_cons_ub.append(du_dt_ub)

    mpc._nlp_cons.pop(3)
    mpc._nlp_cons_lb.pop(3)
    mpc._nlp_cons_ub.pop(3)

    # Scale the objective function
    mpc._nlp_obj *= 10e-2
    
    mpc.create_nlp()

    return mpc


if __name__ == "__main__":
    model = build_benchmark_MPC_model()
    mpc = build_benchmark_MPC(model)