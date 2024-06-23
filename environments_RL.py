import numpy as np
import casadi as cd

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
import control

from collections import deque 

from dataclasses import dataclass
# NOTE: @Mazen I have commented the functions such that you can see directly where which steps are necessary.
# The key function is the step function. In there almost everything is done.
# It could also be the case that you have to adapt the initialization a bit to differ the RL state from the actual state.

@dataclass
class flags:
    system_set: bool = False
    integrator_set: bool = False
    ic_set: bool = False
    stage_cost_set: bool = False
    init_step: bool = False

@dataclass
class integration_settings:
    dt: float = 30.0/3600.0
    opts: dict = None

class Poly_reactor_SB3(gym.Env):

    
    def __init__(self, seed: int = 1234):
        
        # Initialize the flags
        self.flags = flags()

        # Initialize the integration settings
        self.integration_settings = integration_settings()

        # Setup the system equations
        x, u = self._setup_system()

        # Setup the recipe parameterization
        theta_p1, theta_p2, parameter_step = self._setup_recipe_parameterization()

        # Combine for RL state
        s = self._combine_for_RL_state(state = x, recipe_params = self.theta, recipe_step = parameter_step)

        # Setup the scaling
        self._setup_bounds()
        self._setup_scaling()
        self._setup_bounds_for_initialization()

        # Setup spaces
        self.action_space = spaces.Box(low = 0, high = 1, dtype = np.float32)
        self.observation_space = spaces.Box(low = np.zeros(self.s.shape) - np.inf, high = np.ones(self.s.shape) + np.inf, dtype = np.float32)

        self.n_states = self.n_x =  self.x.shape[0]
        self.n_action = self.n_u = self.u.shape[0]

        # Setup the stage cost
        self._setup_stage_cost()

        # Initialize the seed to get reproducible results
        self.seed: int = seed
        self.rng = np.random.default_rng(seed = seed)

        # Initialize the time
        self.time: float = 0.0      # NOTE: In hours
        self.max_time: float = 4.5  # NOTE: In hours
        
        # Initialize the data structure
        self.x_data = np.empty((0, self.n_states))
        self.u_data = np.empty((0, self.n_action))



    def _setup_system(self):

        # System equations
        x, u = self._setup_system_equations()

        # idas
        self._setup_idas()

        return x, u
    


    def _setup_system_equations(self):
        # Certain parameters
        R           = 8.314    			#gas constant
        T_F         = 25 + 273.15       #feed temperature
        E_a         = 8500.0     			#activation energy
        delH_R      = 950.0*1.00      			#sp reaction enthalpy
        A_tank      = 65.0       			#area heat exchanger surface jacket 65

        k_0         = 7.0*1.00      	#sp reaction rate
        k_U2        = 32.0     	#reaction parameter 1
        k_U1        = 4.0      	#reaction parameter 2
        w_WF        = .333      #mass fraction water in feed
        w_AF        = .667      #mass fraction of A in feed

        m_M_KW      = 5000.0      #mass of coolant in jacket
        fm_M_KW     = 300000.0    #coolant flow in jacket 300000;
        m_AWT_KW    = 1000.0      #mass of coolant in EHE
        fm_AWT_KW   = 100000.0    #coolant flow in EHE
        m_AWT       = 200.0       #mass of product in EHE
        fm_AWT      = 20000.0     #product flow in EHE
        m_S         = 39000.0     #mass of reactor steel

        c_pW        = 4.2      #sp heat cap coolant
        c_pS        = .47       #sp heat cap steel
        c_pF        = 3.0         #sp heat cap feed
        self.c_pR        = 5.0         #sp heat cap reactor contents

        k_WS        = 17280.0     #heat transfer coeff water-steel
        k_AS        = 3600.0      #heat transfer coeff monomer-steel
        k_PS        = 360.0       #heat transfer coeff product-steel

        alfa        = 5*20e4*3.6

        p_1         = 1.0

        self.delH_R = 950.0
        k_0 =   7.0

        # States:
        m_W = cd.SX.sym('m_W')                  # NOTE: In kg
        m_A = cd.SX.sym('m_A')                  # NOTE: In kg
        m_P = cd.SX.sym('m_P')                  # NOTE: In kg

        T_R = cd.SX.sym('T_R')                  # NOTE: In K
        T_S = cd.SX.sym('T_S')                  # NOTE: In K
        Tout_M = cd.SX.sym('Tout_M')            # NOTE: In K

        T_EK = cd.SX.sym('T_EK')                # NOTE: In K
        Tout_AWT = cd.SX.sym('Tout_AWT')        # NOTE: In K

        accum_monom = cd.SX.sym('accum_monom')  # NOTE: In kg
        T_adiab = cd.SX.sym('T_adiab')          # NOTE: In K

        self.x = cd.vertcat(m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_monom, T_adiab)
        # self.x_next = cd.SX.sym("x_next", self.x.shape)

        self.x_scaling = cd.DM([10e3, 10e3, 10e3, 1e2, 1e2, 1e2, 1e2, 1e2, 10e3, 1e2])

        # Action:
        m_dot_f = cd.SX.sym('m_dot_f')
        T_in_M =  cd.SX.sym('T_in_M')
        T_in_EK = cd.SX.sym('T_in_EK')

        self.u = cd.vertcat(m_dot_f, T_in_M, T_in_EK)

        # algebraic equations (Just helping expressions)
        U_m    = m_P / (m_A + m_P)
        m_ges  = m_W + m_A + m_P
        k_R1   = k_0 * cd.exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_R2   = k_0 * cd.exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)
        m_A_R  = m_A * (1 - m_AWT / m_ges)


        # Differential equations
        dot_m_W = m_dot_f * w_WF
        dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
        dot_m_P = k_R1 * m_A_R + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)

        dot_T_R = 1./(self.c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * self.c_pR * (T_R - T_EK)) + (self.delH_R * k_R1 * m_A_R))
        dot_T_S =  1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)))
        dot_Tout_M = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)))
        
        dot_T_EK = 1./(self.c_pR * m_AWT)   * ((fm_AWT * self.c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * self.delH_R))
        dot_Tout_AWT = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))
        
        dot_accum_monom = m_dot_f
        dot_T_adiab = self.delH_R/(m_ges*self.c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*self.delH_R/(m_ges*m_ges*self.c_pR))+dot_T_R

        self.rhs = cd.vertcat(dot_m_W, dot_m_A, dot_m_P, dot_T_R, dot_T_S, dot_Tout_M, dot_T_EK, dot_Tout_AWT, dot_accum_monom, dot_T_adiab)
        

        self.flags.system_set = True
        return self.x, self.u
    
    def _setup_recipe_parameterization(self):
        # Recipe parameters

        # This is phase 1
        P1_m_dot_f_slope = cd.SX.sym("P1_m_dot_f_slope")                    # Theta 1
        P1_m_dot_f_yIntersect = cd.SX.sym("P1_m_dot_f_yIntersect")          # Theta 2
        P1_T_R_set = cd.SX.sym("P1_T_R_set")                                # Theta 3
        P1_T_jacket_in_set = cd.SX.sym("P1_T_jacket_in_set")                # Theta 4
        P1_T_EHE_in_set = cd.SX.sym("P1_T_EHE_in_set")                      # Theta 5
        P1_duration = cd.SX.sym("P1_duration")                              # Theta 6
        self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_m_dot_f_yIntersect, P1_T_R_set, P1_T_jacket_in_set, P1_T_EHE_in_set, P1_duration)

        # This is phase 2
        P2_m_dot_f_slope = cd.SX.sym("P2_m_dot_f_slope")                    # Theta 1
        P2_m_dot_f_yIntersect = cd.SX.sym("P2_m_dot_f_yIntersect")          # Theta 2
        P2_T_R_set = cd.SX.sym("P2_T_R_set")                                # Theta 3
        P2_T_jacket_in_set = cd.SX.sym("P2_T_jacket_in_set")                # Theta 4
        P2_duration = cd.SX.sym("P2_duration")                              # Theta 5
        self.theta_p2 = cd.vertcat(P2_m_dot_f_slope, P2_m_dot_f_yIntersect, P2_T_R_set, P2_T_jacket_in_set, P2_duration)

        # This is phase 3
        P3_T_R_set = cd.SX.sym("P3_T_R_set")                                # Theta 1
        P3_T_jacket_in_set = cd.SX.sym("P3_T_jacket_in_set")                # Theta 2
        self.theta_p3 = cd.vertcat(P3_T_R_set, P3_T_jacket_in_set)

        # Combine all thetas
        self.theta = cd.vertcat(self.theta_p1, self.theta_p2, self.theta_p3)
        self.theta_num = cd.DM.zeros(self.theta.shape)
        
        self.theta_scaled = cd.SX.sym("theta_scaled", self.theta.shape)
        self.theta_scaled_num = cd.DM.zeros(self.theta_scaled.shape)

        # Theta Mapping
        self.theta_p1_func = cd.Function("theta_p1_func", [self.theta], [self.theta_p1], ["theta"], ["theta_p1"])
        self.theta_p2_func = cd.Function("theta_p2_func", [self.theta], [self.theta_p2], ["theta"], ["theta_p2"])
        self.theta_p3_func = cd.Function("theta_p3_func", [self.theta], [self.theta_p3], ["theta"], ["theta_p3"])

        
        # Current step
        self.parameter_step = cd.SX.sym("parameter_step")
        self.parameter_step_num = cd.DM.zeros(self.parameter_step.shape)

        self.parameter_step_scaled = cd.SX.sym("parameter_step_scaled", self.parameter_step.shape)
        self.parameter_step_scaled_num = cd.DM.zeros(self.parameter_step_scaled.shape)


        return self.theta_p1, self.theta_p2, self.theta_p3, self.parameter_step
    
    def _setup_bounds(self):     

        # Setup the scaling
        self.x_lower_bounds = np.array([
            0.0e3,
            0.0e3,
            0.0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            0.0e3,
            273.15 + 90 - 2.0
            ]).reshape(-1,1)
        
        self.x_upper_bounds = np.array([
            np.inf,
            np.inf,
            np.inf,
            273.15 + 90.0 + 2.0,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            30e3,
            109. + 273.15,
            ]).reshape(-1,1)


        # As we clip everything this should not be of any issue
        self.u_lower_bounds = np.array([0., 273.15 + 60., 273.15 + 60.]).reshape(-1,1)
        self.u_upper_bounds = np.array([30e3, 273.15 + 100., 273.15 + 100.]).reshape(-1,1)

        # Set the bounds for the recipe parameters
        # First period
        self.theta_p1_lower_bounds = np.array([
            30e3,
            0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 60.0,
            273.15 + 60.0,
            0.5
        ]).reshape(-1,1)
        self.theta_p1_upper_bounds = np.array([
            120e3,
            30e3,
            273.15 + 90.0 + 2.0,
            273.15 + 100,
            273.15 + 100,
            1.5,
        ]).reshape(-1,1)

        # Second period
        self.theta_p2_lower_bounds = np.array([
            -5e3,
            15e3,
            273.15 + 90.0 - 2.0,
            273.15 + 60,
            0.5
        ]).reshape(-1,1)
        self.theta_p2_upper_bounds = np.array([
            0e3,
            30e3,
            273.15 + 90.0 + 2.0,
            273.15 + 100,
            1.5,
        ]).reshape(-1,1)

        # Third period
        self.theta_p3_lower_bounds = np.array([
            273.15 + 90.0 - 2.0,
            273.15 + 60,
        ]).reshape(-1,1)
        self.theta_p3_upper_bounds = np.array([
            273.15 + 90.0 + 2.0,
            273.15 + 100,
        ]).reshape(-1,1)


        self.theta_lower_bounds = np.vstack([self.theta_p1_lower_bounds, self.theta_p2_lower_bounds, self.theta_p3_lower_bounds])
        self.theta_upper_bounds = np.vstack([self.theta_p1_upper_bounds, self.theta_p2_upper_bounds, self.theta_p3_upper_bounds])

        self.parameter_step_lower_bounds = np.array([0.0]).reshape(-1, 1)
        self.parameter_step_upper_bounds = np.array([self.theta.shape[0]]).reshape(-1, 1)

        # self.s_lower = self.s_func(self.x_lower, self.u_lower, self.theta_lower, self.parameter_step_lower)
        # self.s_upper = self.s_func(self.x_upper, self.u_upper, self.theta_upper, self.parameter_step_upper)
        self.s_lower_bounds = self.s_func(self.x_lower_bounds, self.theta_lower_bounds, self.parameter_step_lower_bounds)
        self.s_upper_bounds = self.s_func(self.x_upper_bounds, self.theta_upper_bounds, self.parameter_step_upper_bounds)
    
    def _setup_scaling(self):
        self.theta_scaling_func = cd.Function("theta_scaling", [self.theta], [(self.theta - self.theta_lower_bounds) / (self.theta_upper_bounds  - self.theta_lower_bounds)], ["theta"], ["theta_scaled"])
        self.theta_unscaling_func = cd.Function("theta_unscaling", [self.theta], [self.theta_lower_bounds + self.theta * (self.theta_upper_bounds - self.theta_lower_bounds)], ["theta_scaled"], ["theta"])
        
        self.observation_scaling_func = cd.Function("observation_scaling", [self.s], [(self.s - self.s_lower_bounds) / (self.s_upper_bounds  - self.s_lower_bounds)], ["s"], ["s_scaled"])
        self.observation_unscaling_func = cd.Function("observation_unscaling", [self.s], [self.s_lower_bounds + self.s * (self.s_upper_bounds - self.s_lower_bounds)], ["s_scaled"], ["s"])

    def _setup_bounds_for_initialization(self):     

        # Setup the scaling
        self.x_lower_bounds_for_init = np.array([
            10.0e3,
            0.0e3,
            0.0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 90 - 5.0,
            273.15 + 90.0 - 5.0,
            273.15 + 90.0 - 5.0,
            273.15 + 90.0 - 5.0,
            0.0e3,
            273.15 + 90 - 2.0
            ]).reshape(-1,1)
        
        self.x_upper_bounds_for_init = np.array([
            15.0e3,
            5.0e3,
            0.0e3,
            273.15 + 90.0 + 2.0,
            273.15 + 95.0,
            273.15 + 95.0,
            273.15 + 95.0,
            273.15 + 95.0,
            0.0e3,
            100. + 273.15,
            ]).reshape(-1,1)



    def _combine_for_RL_state(self, **kwargs):
        

        self.s = cd.vertcat(*[value for value in kwargs.values()])
        self.s_func = cd.Function("s", [*[value for value in kwargs.values()]], [self.s], [*[keys for keys in kwargs.keys()]], ["s"])
        self.rev_s = cd.Function("rev_s", [self.s], [*[value for value in kwargs.values()]], ["s"], [*[keys for keys in kwargs.keys()]])

        return self.s
    
    def _setup_idas(self):

        self.dae_dict = {
            "x": self.x,
            "p": self.u,
            "ode": self.rhs,
        }

        if self.integration_settings.opts is None:
            self.integration_settings.opts = {
                # "rf": "kinsol",
                # "rootfinder_options": {"strategy": "linesearch"}
                # "verbose": True,
            }

        # Build the integrator
        self.integrator = cd.integrator("system", "idas", self.dae_dict, 0.0, self.integration_settings.dt, self.integration_settings.opts)
        
        # Simplify the integration in- and outputs
        integrator_inputs = self.integrator.name_in()[:3]
        intgrator_outputs = self.integrator.name_out()[:2]
        self.integrator = self.integrator.factory("system", integrator_inputs, intgrator_outputs)

        self.flags.integrator_set = True
        return
    
    def _setup_stage_cost(self):

        self.relaxation_weights_x = cd.DM.ones(self.x.shape) * 1e5
        self.relaxation_weights_u = cd.DM.ones(self.u.shape) * 1e5

        # Maximize the product of the reactor 
        self.stage_cost = 20e3 - self.x[2] # Maybe we have to look at this again

        # Penalize constraint violations
        self.stage_cost_penalties = cd.fmax(self.x_lower_bounds - self.x, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.x - self.x_upper_bounds, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.u_lower_bounds - self.u, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u
        self.stage_cost_penalties += cd.fmax(self.u - self.u_upper_bounds, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u

        self.stage_cost += self.stage_cost_penalties

        # Scaling
        self.stage_cost *= 1 / 20e3 
        self.stage_cost_penalties *= 1 / 20e3 

        # self.stage_cost = cd.log10(self.stage_cost)
        # self.stage_cost_penalties = cd.log10(self.stage_cost_penalties)

        # Adapting to step length
        self.stage_cost *= self.integration_settings.dt
        self.stage_cost_penalties *= self.integration_settings.dt


        self._stage_cost_func = cd.Function("stage_cost", [self.x, self.u], [self.stage_cost], ["x", "u",], ["stage_cost"])
        return
    

    def _truncated_check(self, state: np.ndarray, action: np.ndarray) -> bool:
        truncation = False

        action_tolerance = 0.1 # Percentage of range from min to max
        delta_a_max = (self.u_upper_bounds - self.u_lower_bounds) * action_tolerance

        if np.any(action - self.u_upper_bounds > delta_a_max) or np.any(self.u_lower_bounds - action > delta_a_max):
            truncation = True
            return truncation

        state_tolerance = 0.1 # Percentage of range from min to max
        delta_s_max = (self.x_upper_bounds - self.x_lower_bounds) * state_tolerance
        delta_s_max[3] = 4.0

        if np.any(state - self.x_upper_bounds > delta_s_max) or np.any(self.x_lower_bounds - state > delta_s_max):
            truncation = True
            return truncation

        return truncation
      
    # This method simulates a hole batch stage i.e. an actual step, dependoing
    # on the batch stage
    def step(self, theta: np.ndarray):

        # Update the theta vecors
        self.theta_scaled_num[self.parameter_step_num] = theta
        self.theta_num = self.theta_unscaling_func(self.theta_scaled_num)

        self.parameter_step_num += 1


        if self.parameter_step_num <= self.theta_p1.shape[0]:
            observation, reward, termination, truncation, info = self._first_stage()

        elif self.parameter_step_num <= self.theta_p1.shape[0] + self.theta_p2.shape[0]:
            observation, reward, termination, truncation, info = self._second_stage()

        else:
            observation, reward, termination, truncation, info = self._third_stage()

        observation = self.observation_scaling_func(observation)

        return observation, reward, termination, truncation, info   
            
            
            
    
    # This method executes the simulation of the first batch stage
    def _first_stage(self):

        if self.parameter_step_num < self.theta_p1.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_first_stage()

        return observation, reward, termination, truncation, info

        
    def _simulate_first_stage(self):
        
        # Extract current time
        t0 = self.time

        # Extract and assign the recipe parameters
        theta_p1_num = self.theta_p1_func(self.theta_num)

        m_dot_f_slope = float(theta_p1_num[0])
        m_dot_f_yIntercept = float(theta_p1_num[1])

        T_R_set = float(theta_p1_num[2])
        T_Jacket_in_num_set = float(theta_p1_num[3])
        T_EHE_cool_in_num = float(theta_p1_num[4])

        max_duration = float(theta_p1_num[5])


        # Define controller variables
        K_P_T_J = 10
        K_I_T_J = 1.0

        K_P_T_EHE = 10
        K_I_T_EHE = 1.0

        integrated_error_T_J = 0
        anti_windup_Z_J = False

        integrated_error_T_EHE = 0
        anti_windup_T_EHE = False

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0


        while self.time < t0 + max_duration and state[-2] < self.x_upper_bounds[-2]:
            
            # Define the control actions
            m_dot_f_num = np.clip(m_dot_f_yIntercept + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = self.u_upper_bounds[0])
            
            error_T_J = T_R_set - state[3]
            integrated_error_T_J += error_T_J * self.integration_settings.dt
            T_Jacket_in_num = T_Jacket_in_num_set + K_P_T_J * error_T_J + K_I_T_J * integrated_error_T_J

            if anti_windup_Z_J:
                integrated_error_T_J = np.clip(integrated_error_T_J, a_min = - np.abs(integrated_anti_windup_error_T_J), a_max = + np.abs(integrated_anti_windup_error_T_J))

            if T_Jacket_in_num <  self.u_lower_bounds[1] or T_Jacket_in_num > self.u_upper_bounds[1]:
                T_Jacket_in_num = np.clip(T_Jacket_in_num, a_min = self.u_lower_bounds[1], a_max = self.u_upper_bounds[1])

                integrated_anti_windup_error_T_J = integrated_error_T_J
                anti_windup_Z_J = True
            else:
                anti_windup_Z_J = False
            

            # T_EHE_cool_in_num = 273.15 + 100.0
            error_T_EHE = T_R_set - state[3]
            integrated_error_T_EHE += error_T_EHE * self.integration_settings.dt
            T_EHE_cool_in_num = T_EHE_cool_in_num + K_P_T_EHE * error_T_EHE + K_I_T_EHE * integrated_error_T_EHE

            if anti_windup_T_EHE:
                integrated_error_T_EHE = np.clip(integrated_error_T_EHE, a_min = - np.abs(integrated_anti_windup_error_T_EHE), a_max = + np.abs(integrated_anti_windup_error_T_EHE))
            
            if T_EHE_cool_in_num <  self.u_lower_bounds[2] or T_EHE_cool_in_num > self.u_upper_bounds[2]:            
                T_EHE_cool_in_num = np.clip(T_EHE_cool_in_num, a_min = self.u_lower_bounds[2], a_max = self.u_upper_bounds[2])

                integrated_anti_windup_error_T_EHE = integrated_error_T_EHE
                anti_windup_T_EHE = True
            else:
                anti_windup_T_EHE = False


            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EHE_cool_in_num)

            state = self.x_num

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True

        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = False
        info = {}

        return observation, reward, termination, global_truncation, info
        
    # This method executes the simulation of the second batch stage
    def _second_stage(self):

        if self.parameter_step_num < self.theta_p1.shape[0] + self.theta_p2.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_second_stage()

        return observation, reward, termination, truncation, info
    
    def _simulate_second_stage(self):

        # Extract current time
        t0 = self.time

        # Extract and assign the recipe parameters
        theta_p2_num = self.theta_p2_func(self.theta_num)

        m_dot_f_slope = float(theta_p2_num[0])
        m_dot_f_yIntercept = float(theta_p2_num[1])

        T_R_set = float(theta_p2_num[2])
        T_Jacket_in_num_set = float(theta_p2_num[3])

        max_duration = float(theta_p2_num[4])


        # Define controller variables
        K_P = 10
        K_I = 1.0

        integrated_error = 0
        anti_windup = False

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0
        
        while self.time < t0 + max_duration and state[-2] < self.x_upper_bounds[-2]:
            
            # Define the control actions
            m_dot_f_num = np.clip(m_dot_f_yIntercept + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = self.u_upper_bounds[0])
            
            error = T_R_set - state[3]
            integrated_error += error * self.integration_settings.dt
            T_Jacket_in_num = T_Jacket_in_num_set + K_P * error + K_I * integrated_error

            if anti_windup:
                integrated_error = np.clip(integrated_error, a_min = - np.abs(integrated_anti_windup_error), a_max = + np.abs(integrated_anti_windup_error))

            if T_Jacket_in_num <  273.15 + 60 or T_Jacket_in_num > 273.15 + 100:
                T_Jacket_in_num = np.clip(T_Jacket_in_num, a_min = 273.15 + 60, a_max = 273.15 + 100)

                integrated_anti_windup_error = integrated_error
                anti_windup = True
            else:
                anti_windup = False
            
            T_EHE_cool_in_num = 273.15 + 60.0

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EHE_cool_in_num)

            state = self.x_num

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True

        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = False
        info = {}

        return observation, reward, termination, global_truncation, info
    
    # This method executes the simulation of the third batch stage    
    def _third_stage(self):
        if self.parameter_step_num < self.theta.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_third_stage()

        return observation, reward, termination, truncation, info

    def _simulate_third_stage(self):
        
        # Extract and assign the recipe parameters
        theta_p3_num = self.theta_p3_func(self.theta_num)

        T_R_set = float(theta_p3_num[0])
        T_Jacket_in_num_set = float(theta_p3_num[1])

        # Define controller variables
        K_P = 10.0
        K_I = 1.0

        integrated_error = 0.0
        anti_windup = False

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0

        while self.time < self.max_time:

            # Define the control actions
            m_dot_f_num = 0.0e3
            
            error = T_R_set - state[3]
            integrated_error += error * self.integration_settings.dt
            T_Jacket_in_num = T_Jacket_in_num_set + K_P * error + K_I * integrated_error

            if anti_windup:
                integrated_error = np.clip(integrated_error, a_min = - np.abs(integrated_anti_windup_error), a_max = + np.abs(integrated_anti_windup_error))

            if T_Jacket_in_num <  273.15 + 60 or T_Jacket_in_num > 273.15 + 100:
                T_Jacket_in_num = np.clip(T_Jacket_in_num, a_min = 273.15 + 60, a_max = 273.15 + 100)

                integrated_anti_windup_error = integrated_error
                anti_windup = True
            else:
                anti_windup = False
            
            T_EHE_cool_in_num = 273.15 + 100.0

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EHE_cool_in_num)

            state = self.x_num

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True
            
        
        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = True
        info = {}

        return observation, reward, termination, global_truncation, info
               

    def _compute_stage_cost(self, state: np.ndarray, action: np.ndarray):
        # Basic stage cost
        stage_cost = float(self._stage_cost_func(state, action).full())
        stage_cost *= -1 # Because RL works with positive rewards (neg stage cost)
        return stage_cost
    
    def _done_check(self, next_state: np.ndarray, old_state: np.ndarray, taken_action: np.ndarray) -> bool:

        # action check
        action_tolerance = np.array([300, 5, 5]).reshape(-1,1)
        if np.any(taken_action < self.u_lower - action_tolerance) or np.any(taken_action > self.u_upper + action_tolerance):
            return True
        
        # NOTE: @Mazen: It could be the case that this is way to tight.
        # We should check this when we have a first draft and first interactions.

        # old state check
        state_tolerance_lower = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,   0.0, 1.0]).reshape(-1,1)
        state_tolerance_upper = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0]).reshape(-1,1)
        if np.any(old_state < self.x_lower - state_tolerance_lower) or np.any(old_state > self.x_upper + state_tolerance_upper):
            return True

        # next state check
        # state_tolerance_lower = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]).reshape(-1,1)
        # state_tolerance_upper = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 100, 10.0]).reshape(-1,1)
        # if np.any(next_state < self.x_lower - state_tolerance_lower) or np.any(next_state > self.x_upper + state_tolerance_upper):
        #     return True

        if self.timestep >= self.max_steps:
            return True
        return False



    def _sample_new_initial_state(self):

        # Sample uniform in the initialization space
        x_num = self.x_lower_bounds_for_init + (self.x_upper_bounds_for_init - self.x_lower_bounds_for_init) * self.rng.uniform(low = 0.0, high = 1.0, size = (self.n_states, 1))
        
        # Set some constraints on the initial state
        x_num[4] = x_num[3].copy() # Steel temp = reactor temp

        # Calculating a feasible T_adiab
        x_num[9] = x_num[3] + self.delH_R / self.c_pR * x_num[1] / (x_num[0] + x_num[1] + x_num[2])

        if x_num[9] > self.x_upper_bounds_for_init[-1]:
            x_num[9] = self.x_upper_bounds_for_init[-1]
            phi = (x_num[9] - x_num[3]) * self.c_pR / self.delH_R
            x_num[1] = phi / (1 - phi) * (x_num[0] + x_num[2])
        self.x_num = x_num.copy() 

        return self.x_num

        

    def reset(self, seed: int = None, options = None):
        
        # Reset the inner state
        self.x_num = self._sample_new_initial_state()

        # Reset the taken action (this is actually not necessary because we do not have a delta u term)
        self.u_num = np.array([0, 273.15 + 60, 273.15 + 60]).reshape(-1,1)

        # Reset the recipe parameters (do it with the scaled value)
        self.theta_scaled_num = cd.DM.zeros(self.theta.shape)
        self.theta_num = self.theta_unscaling_func(self.theta_scaled_num)
        self.parameter_step_num = 0

        # Stack everything to the whole observation
        # observation = self.s_num = self.s_func(self.x_num, self.u_num, self.theta_num, self.parameter_step_num)
        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)

        # Make sure that the agent just sees the scaled version
        observation = self.observation_scaling_func(observation)

        # Reset the time
        self.time = 0.0

        # Reset the data because it will blow up otherwise
        self.x_data = np.empty((0, self.x_num.shape[0]))
        self.u_data = np.empty((0, self.u_num.shape[0]))

        # Say that everythin is initialized now
        self.flags.ic_set = True

        return observation, {}





class Poly_reactor_SB3_cascade(Poly_reactor_SB3):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._setup_controllers()
        
        self.setpoint_data = np.empty((0, 3))

        self.violations_list = deque(maxlen = 50)
        self.violations: int = 0

    def _setup_system_equations(self):
        # Certain parameters
        R           = 8.314    			#gas constant
        T_F         = 25 + 273.15       #feed temperature
        E_a         = 8500.0     			#activation energy
        delH_R      = 950.0*1.00      			#sp reaction enthalpy
        A_tank      = 65.0       			#area heat exchanger surface jacket 65

        k_0         = 7.0*1.00      	#sp reaction rate
        k_U2        = 32.0     	#reaction parameter 1
        k_U1        = 4.0      	#reaction parameter 2
        w_WF        = .333      #mass fraction water in feed
        self.w_AF = w_AF        = .667      #mass fraction of A in feed

        m_M_KW      = 5000.0      #mass of coolant in jacket
        fm_M_KW     = 300000.0    #coolant flow in jacket 300000;
        m_AWT_KW    = 1000.0      #mass of coolant in EHE
        fm_AWT_KW   = 100000.0    #coolant flow in EHE
        self.m_AWT = m_AWT       = 200.0       #mass of product in EHE
        fm_AWT      = 20000.0     #product flow in EHE
        m_S         = 39000.0     #mass of reactor steel

        c_pW        = 4.2      #sp heat cap coolant
        c_pS        = .47       #sp heat cap steel
        c_pF        = 3.0         #sp heat cap feed
        self.c_pR        = 5.0         #sp heat cap reactor contents

        self.k_WS = k_WS        = 17280.0     #heat transfer coeff water-steel
        k_AS        = 3600.0      #heat transfer coeff monomer-steel
        k_PS        = 360.0       #heat transfer coeff product-steel

        alfa        = 5*20e4*3.6

        p_1         = 1.0

        self.delH_R = 950.0
        k_0 =   7.0

        # States:
        m_W = cd.SX.sym('m_W')                  # NOTE: In kg
        m_A = cd.SX.sym('m_A')                  # NOTE: In kg
        m_P = cd.SX.sym('m_P')                  # NOTE: In kg

        T_R = cd.SX.sym('T_R')                  # NOTE: In K
        T_S = cd.SX.sym('T_S')                  # NOTE: In K
        Tout_M = cd.SX.sym('Tout_M')            # NOTE: In K

        T_EK = cd.SX.sym('T_EK')                # NOTE: In K
        Tout_AWT = cd.SX.sym('Tout_AWT')        # NOTE: In K

        accum_monom = cd.SX.sym('accum_monom')  # NOTE: In kg
        T_adiab = cd.SX.sym('T_adiab')          # NOTE: In K

        self.x = cd.vertcat(m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_monom, T_adiab)
        # self.x_next = cd.SX.sym("x_next", self.x.shape)

        self.x_scaling = cd.DM([10e3, 10e3, 10e3, 1e2, 1e2, 1e2, 1e2, 1e2, 10e3, 1e2])

        # Action:
        m_dot_f = cd.SX.sym('m_dot_f')
        T_in_M =  cd.SX.sym('T_in_M')
        T_in_EK = cd.SX.sym('T_in_EK')

        self.u = cd.vertcat(m_dot_f, T_in_M, T_in_EK)

        # algebraic equations (Just helping expressions)
        U_m    = m_P / (m_A + m_P)
        m_ges  = m_W + m_A + m_P
        k_R1   = k_0 * cd.exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_R2   = k_0 * cd.exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)
        m_A_R  = m_A * (1 - m_AWT / m_ges)


        # Differential equations
        dot_m_W = m_dot_f * w_WF
        dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
        dot_m_P = k_R1 * m_A_R + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)

        dot_T_R = 1./(self.c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * self.c_pR * (T_R - T_EK)) + (self.delH_R * k_R1 * m_A_R))
        dot_T_S =  1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)))
        dot_Tout_M = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)))
        
        dot_T_EK = 1./(self.c_pR * m_AWT)   * ((fm_AWT * self.c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * self.delH_R))
        dot_Tout_AWT = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))
        
        dot_accum_monom = m_dot_f
        dot_T_adiab = self.delH_R/(m_ges*self.c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*self.delH_R/(m_ges*m_ges*self.c_pR))+dot_T_R

        self.rhs = cd.vertcat(dot_m_W, dot_m_A, dot_m_P, dot_T_R, dot_T_S, dot_Tout_M, dot_T_EK, dot_Tout_AWT, dot_accum_monom, dot_T_adiab)
        

        self.flags.system_set = True


        # Simplified system for cascade control
        m_ges_lumped = cd.SX.sym('m_ges_lumped')
        x_lumped = cd.vertcat(T_R, Tout_M, T_EK, Tout_AWT, m_ges_lumped)
        u_lumped = cd.vertcat(m_dot_f, T_in_M, T_in_EK)

        dot_T_R_lumped = 1./(self.c_pR * m_ges_lumped)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_WS *A_tank* (T_R - Tout_M)) - (fm_AWT * self.c_pR * (T_R - T_EK)))
        dot_Tout_M_lumped = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_WS *A_tank* (T_R - Tout_M)))
        dot_T_EK_lumped = 1./(self.c_pR * m_AWT)   * ((fm_AWT * self.c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)))
        dot_Tout_AWT_lumped = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))
        dot_m_ges_lumped = m_dot_f

        self.rhs_lumped = cd.vertcat(dot_T_R_lumped, dot_Tout_M_lumped, dot_T_EK_lumped, dot_Tout_AWT_lumped, dot_m_ges_lumped)
        self.rhs_lumped_func = cd.Function("rhs_lumped", [x_lumped, u_lumped], [self.rhs_lumped], ["x", "u"], ["rhs_lumped"])
        self.A_mat_lumped = cd.jacobian(self.rhs_lumped, x_lumped)
        self.B_mat_lumped = cd.jacobian(self.rhs_lumped, u_lumped)
        self.A_mat_lumped_func = cd.Function("A_mat_lumped", [x_lumped, u_lumped], [self.A_mat_lumped], ["x", "u"], ["A_mat_lumped"])
        self.B_mat_lumped_func = cd.Function("B_mat_lumped", [x_lumped, u_lumped], [self.B_mat_lumped], ["x", "u"], ["B_mat_lumped"])


        return self.x, self.u
    
    def _setup_recipe_parameterization(self):
        # Recipe parameters

        # This is phase 0
        P0_m_dot_f = cd.SX.sym("P0_m_dot_f")                                # Theta 1
        P0_T_R_set = cd.SX.sym("P0_T_R_set")                                # Theta 2
        P0_K_P = cd.SX.sym("P0_K_P")                                        # Theta 2
        P0_K_I = cd.SX.sym("P0_K_I")                                        # Theta 3
        # P0_duration = cd.SX.sym("P0_duration")                              # Theta 2
        P0_m_tot_min = cd.SX.sym("P0_m_tot_min")                              # Theta 2
        self.theta_p0 = cd.vertcat(P0_m_dot_f, P0_T_R_set, P0_K_P, P0_K_I, P0_m_tot_min)

        # This is phase 1
        P1_m_dot_f_slope = cd.SX.sym("P1_m_dot_f_slope")                    # Theta 1
        # P1_m_dot_f_yIntersect = cd.SX.sym("P1_m_dot_f_yIntersect")          # Theta 2
        P1_T_R_set = cd.SX.sym("P1_T_R_set")                                # Theta 3
        P1_K_P = cd.SX.sym("P1_K_P")                                          # Theta 4
        P1_K_I = cd.SX.sym("P1_K_I")                                          # Theta 5
        # P1_T_jacket_set = cd.SX.sym("P1_T_jacket_set")                      # Theta 4
        # P1_T_EHE_set = cd.SX.sym("P1_T_EHE_set")                            # Theta 5
        # P1_T_jacket_in_set = cd.SX.sym("P1_T_jacket_in_set")                # Theta 6
        # P1_T_EHE_in_set = cd.SX.sym("P1_T_EHE_in_set")                      # Theta 7
        P1_duration = cd.SX.sym("P1_duration")                              # Theta 8
        P1_m_dot_f_max = cd.SX.sym("P1_m_dot_f_max")                        # Theta 9
        # self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_m_dot_f_yIntersect, P1_T_R_set, P1_T_jacket_set, P1_T_EHE_set, P1_T_jacket_in_set, P1_T_EHE_in_set, P1_duration)
        # self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_T_R_set, P1_T_jacket_set, P1_T_EHE_set, P1_T_jacket_in_set, P1_T_EHE_in_set, P1_duration)
        # self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_T_R_set, P1_T_jacket_set, P1_T_EHE_set, P1_T_jacket_in_set, P1_T_EHE_in_set, P1_duration, P1_m_dot_f_max)
        # self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_T_R_set, P1_T_jacket_set, P1_T_EHE_set, P1_duration, P1_m_dot_f_max)
        self.theta_p1 = cd.vertcat(P1_m_dot_f_slope, P1_T_R_set, P1_K_P, P1_K_I, P1_duration, P1_m_dot_f_max)


        # This is phase 2
        # P2_m_dot_f_slope = cd.SX.sym("P2_m_dot_f_slope")                    # Theta 1
        # P2_m_dot_f_yIntersect = cd.SX.sym("P2_m_dot_f_yIntersect")          # Theta 2
        P2_T_R_set = cd.SX.sym("P2_T_R_set")                                # Theta 3
        P2_K_P = cd.SX.sym("P2_K_P")                                        # Theta 4
        P2_K_I = cd.SX.sym("P2_K_I")                                        # Theta 5
        # P2_T_jacket_set = cd.SX.sym("P2_T_jacket_set")                      # Theta 4
        # P2_T_EHE_set = cd.SX.sym("P2_T_EHE_set")                            # Theta 5
        # P2_T_jacket_in_set = cd.SX.sym("P2_T_jacket_in_set")                # Theta 6
        # P2_T_EHE_in_set = cd.SX.sym("P2_T_EHE_in_set")                      # Theta 7
        # P2_duration = cd.SX.sym("P2_duration")                              # Theta 8
        # self.theta_p2 = cd.vertcat(P2_m_dot_f_slope, P2_m_dot_f_yIntersect, P2_T_R_set, P2_T_jacket_set, P2_T_EHE_set, P2_T_jacket_in_set, P2_T_EHE_in_set, P2_duration)
        # self.theta_p2 = cd.vertcat(P2_m_dot_f_yIntersect, P2_T_R_set, P2_T_jacket_set, P2_T_EHE_set, P2_T_jacket_in_set, P2_T_EHE_in_set)
        # self.theta_p2 = cd.vertcat(P2_m_dot_f_slope, P2_m_dot_f_yIntersect, P2_T_R_set, P2_T_jacket_set, P2_T_EHE_set)
        # self.theta_p2 = cd.vertcat(P2_m_dot_f_slope, P2_m_dot_f_yIntersect, P2_T_R_set)
        self.theta_p2 = cd.vertcat(P2_T_R_set, P2_K_P, P2_K_I)

        # This is phase 3
        # P3_T_R_set = cd.SX.sym("P3_T_R_set")                                # Theta 1
        # P3_T_jacket_set = cd.SX.sym("P3_T_jacket_set")                      # Theta 2
        # P3_T_EHE_set = cd.SX.sym("P3_T_EHE_set")                            # Theta 3
        # P3_T_jacket_in_set = cd.SX.sym("P3_T_jacket_in_set")                # Theta 4
        # P3_T_EHE_in_set = cd.SX.sym("P3_T_EHE_in_set")                      # Theta 5
        # self.theta_p3 = cd.vertcat(P3_T_R_set, P3_T_jacket_set, P3_T_EHE_set, P3_T_jacket_in_set, P3_T_EHE_in_set)

        # Combine all thetas
        # self.theta = cd.vertcat(self.theta_p1, self.theta_p2, self.theta_p3)
        # self.theta = cd.vertcat(self.theta_p0, self.theta_p1, self.theta_p2, self.theta_p3)
        self.theta = cd.vertcat(self.theta_p0, self.theta_p1, self.theta_p2)
        # self.theta = cd.vertcat(self.theta_p1, self.theta_p2)
        self.theta_num = cd.DM.zeros(self.theta.shape)
        
        self.theta_scaled = cd.SX.sym("theta_scaled", self.theta.shape)
        self.theta_scaled_num = cd.DM.zeros(self.theta_scaled.shape)

        # Theta Mapping
        self.theta_p0_func = cd.Function("theta_p0_func", [self.theta], [self.theta_p0], ["theta"], ["theta_p0"])
        self.theta_p1_func = cd.Function("theta_p1_func", [self.theta], [self.theta_p1], ["theta"], ["theta_p1"])
        self.theta_p2_func = cd.Function("theta_p2_func", [self.theta], [self.theta_p2], ["theta"], ["theta_p2"])
        # self.theta_p3_func = cd.Function("theta_p3_func", [self.theta], [self.theta_p3], ["theta"], ["theta_p3"])

        
        # Current step
        self.parameter_step = cd.SX.sym("parameter_step")
        self.parameter_step_num = cd.DM.zeros(self.parameter_step.shape)

        self.parameter_step_scaled = cd.SX.sym("parameter_step_scaled", self.parameter_step.shape)
        self.parameter_step_scaled_num = cd.DM.zeros(self.parameter_step_scaled.shape)


        return self.theta_p1, self.theta_p2,  self.parameter_step
    
    def _setup_bounds(self):     

        # Setup the scaling
        self.x_lower_bounds = np.array([
            0.0e3,
            0.0e3,
            0.0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            0.0e3,
            273.15 + 90 - 2.0
            ]).reshape(-1,1)
        
        self.x_upper_bounds = np.array([
            30e3, #np.inf,
            30e3, #np.inf,
            30e3, #np.inf,
            273.15 + 90.0 + 2.0,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            30e3,
            109. + 273.15,
            ]).reshape(-1,1)


        # As we clip everything this should not be of any issue
        self.u_lower_bounds = np.array([0., 273.15 + 60., 273.15 + 60.]).reshape(-1,1)
        self.u_upper_bounds = np.array([30e3, 273.15 + 100., 273.15 + 100.]).reshape(-1,1)

        # Set the bounds for the recipe parameters
        # Zeroth period
        self.theta_p0_lower_bounds = np.array([
            0.1e3,
            273.15 + 90.0 - 2.0,
            0.0,
            1e-3,
            10e3,
        ]).reshape(-1,1)
        self.theta_p0_upper_bounds = np.array([
            30e3,
            273.15 + 90.0 + 2.0,
            10.0,
            100.0,
            40e3,
        ]).reshape(-1,1)


        # First period
        # self.theta_p1_lower_bounds = np.array([
        #     0e3,
        #     0e3,
        #     273.15 + 90.0 - 2.0,
        #     273.15 + 60.0,
        #     273.15 + 60.0,
        #     273.15 + 60.0,
        #     273.15 + 60.0,
        #     0.5
        # ]).reshape(-1,1)
        # self.theta_p1_upper_bounds = np.array([
        #     60e3,
        #     30e3,
        #     273.15 + 90.0 + 2.0,
        #     273.15 + 100,
        #     273.15 + 100,
        #     273.15 + 100,
        #     273.15 + 100,
        #     1.5,
        # ]).reshape(-1,1)

        self.theta_p1_lower_bounds = np.array([
            5e3,
            273.15 + 90.0 - 2.0,
            0.0,
            1e-3,
            # 273.15 + 85.0,
            # 273.15 + 85.0,
            # 273.15 + 60.0,
            # 273.15 + 60.0,
            0.0,
            5e3,
        ]).reshape(-1,1)
        self.theta_p1_upper_bounds = np.array([
            30e3,
            273.15 + 90.0 + 2.0,
            20.0,
            100.0,
            # 273.15 + 95,
            # 273.15 + 95,
            # 273.15 + 100,
            # 273.15 + 100,
            3.0,
            30e3,
        ]).reshape(-1,1)

        # Second period
        # self.theta_p2_lower_bounds = np.array([
        #     -5e3,
        #     15e3,
        #     273.15 + 90.0 - 2.0 + 0.5,
        #     273.15 + 85.0,
        #     273.15 + 85.0,
        #     273.15 + 60.0,
        #     273.15 + 60.0,
        #     0.5
        # ]).reshape(-1,1)
        # self.theta_p2_upper_bounds = np.array([
        #     0e3,
        #     30e3,
        #     273.15 + 90.0 + 2.0 - 0.5,
        #     273.15 + 95,
        #     273.15 + 95,
        #     273.15 + 100,
        #     273.15 + 100,
        #     2.0,
        # ]).reshape(-1,1)
        self.theta_p2_lower_bounds = np.array([
            # -5e3,
            # 10e3,
            273.15 + 90.0 - 2.0,
            0.0,
            1e-3,
            # 273.15 + 85.0,
            # 273.15 + 85.0,
            # 273.15 + 60.0,
            # 273.15 + 60.0,
        ]).reshape(-1,1)
        self.theta_p2_upper_bounds = np.array([
            # 0,
            # 20e3,
            273.15 + 90.0 + 2.0,
            30.0,
            150.0,
            # 273.15 + 95,
            # 273.15 + 95,
            # 273.15 + 100,
            # 273.15 + 100,
        ]).reshape(-1,1)

        # Third period
        # self.theta_p3_lower_bounds = np.array([
        #     273.15 + 90.0 - 2.0 + 0.5,
        #     273.15 + 85.0,
        #     273.15 + 85.0,
        #     273.15 + 60.0,
        #     273.15 + 60.0,
        # ]).reshape(-1,1)
        # self.theta_p3_upper_bounds = np.array([
        #     273.15 + 90.0 + 2.0 - 0.5,
        #     273.15 + 95,
        #     273.15 + 95,
        #     273.15 + 100,
        #     273.15 + 100,
        # ]).reshape(-1,1)


        # self.theta_lower_bounds = np.vstack([self.theta_p1_lower_bounds, self.theta_p2_lower_bounds, self.theta_p3_lower_bounds])
        # self.theta_upper_bounds = np.vstack([self.theta_p1_upper_bounds, self.theta_p2_upper_bounds, self.theta_p3_upper_bounds])

        # self.theta_lower_bounds = np.vstack([self.theta_p0_lower_bounds, self.theta_p1_lower_bounds, self.theta_p2_lower_bounds, self.theta_p3_lower_bounds])
        # self.theta_upper_bounds = np.vstack([self.theta_p0_upper_bounds, self.theta_p1_upper_bounds, self.theta_p2_upper_bounds, self.theta_p3_upper_bounds])

        self.theta_lower_bounds = np.vstack([self.theta_p0_lower_bounds, self.theta_p1_lower_bounds, self.theta_p2_lower_bounds])
        self.theta_upper_bounds = np.vstack([self.theta_p0_upper_bounds, self.theta_p1_upper_bounds, self.theta_p2_upper_bounds])
        # self.theta_lower_bounds = np.vstack([self.theta_p1_lower_bounds, self.theta_p2_lower_bounds])
        # self.theta_upper_bounds = np.vstack([self.theta_p1_upper_bounds, self.theta_p2_upper_bounds])

        self.parameter_step_lower_bounds = np.array([0.0]).reshape(-1, 1)
        self.parameter_step_upper_bounds = np.array([self.theta.shape[0]]).reshape(-1, 1)

        # self.s_lower = self.s_func(self.x_lower, self.u_lower, self.theta_lower, self.parameter_step_lower)
        # self.s_upper = self.s_func(self.x_upper, self.u_upper, self.theta_upper, self.parameter_step_upper)
        self.s_lower_bounds = self.s_func(self.x_lower_bounds, self.theta_lower_bounds, self.parameter_step_lower_bounds)
        self.s_upper_bounds = self.s_func(self.x_upper_bounds, self.theta_upper_bounds, self.parameter_step_upper_bounds)


    def _reward_tracking(self, state: np.ndarray, T_R_set: float, delta_u: np.ndarray):
        # Calculate the reward
        reward = - (state[3]- T_R_set ) ** 2 * self.integration_settings.dt

        R = cd.diag([0.1, 0.1])
        # R = cd.diag([1., 1.])
        reward += - delta_u.T @ R @ delta_u * self.integration_settings.dt
        return reward
    
    def _setup_controllers(self):
        self.controllers_p0 = {}
        self.controllers_p0["outer_T_J"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 5, K_I = 20, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        self.controllers_p0["outer_T_EK"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 5, K_I = 20, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        self.controllers_p0["inner_T_J_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        self.controllers_p0["inner_T_EK_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])


        # Controller for period 1
        self.controllers_p1 = {}

        self.controllers_p1["outer_T_J"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1, K_I = 10, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        self.controllers_p1["outer_T_EK"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1, K_I = 10, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        self.controllers_p1["inner_T_J_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        self.controllers_p1["inner_T_EK_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 0.9, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])

        # Controller for period 2
        self.controllers_p2 = {}

        self.controllers_p2["outer_T_J"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 6, K_I = 20, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        self.controllers_p2["outer_T_EK"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 6, K_I = 20, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        self.controllers_p2["inner_T_J_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 0.5, K_I = 15, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        self.controllers_p2["inner_T_EK_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 0.5, K_I = 15, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])        

        # Controller for period 3
        self.controllers_p3 = {}

        self.controllers_p3["outer_T_J"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 15, K_I = 50, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        self.controllers_p3["outer_T_EK"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 15, K_I = 50, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        # self.controllers_p3["inner_T_J_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        # self.controllers_p3["inner_T_EK_in"] = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = 273.15 + 90, K_P = 0.9, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])

        return
    
    def step(self, theta: np.ndarray):

        # Update the theta vecors
        self.theta_scaled_num[self.parameter_step_num] = theta
        self.theta_num = self.theta_unscaling_func(self.theta_scaled_num)

        self.parameter_step_num += 1

        if self.parameter_step_num <= self.theta_p0.shape[0]:
            observation, reward, termination, truncation, info = self._zeroth_stage()

        elif self.parameter_step_num <= self.theta_p0.shape[0] + self.theta_p1.shape[0]:
        # if self.parameter_step_num <= self.theta_p1.shape[0]:
            observation, reward, termination, truncation, info = self._first_stage()

        # elif self.parameter_step_num <= self.theta_p0.shape[0] + self.theta_p1.shape[0] + self.theta_p2.shape[0]:
        else:
            observation, reward, termination, truncation, info = self._second_stage()

        # else:
        #     observation, reward, termination, truncation, info = self._third_stage()

        observation = self.observation_scaling_func(observation)

        return observation, reward, termination, truncation, info   
    
    def _violation_check(self, state: np.ndarray, action: np.array):
        # if np.any(action - self.u_upper_bounds > 0.0) or np.any(self.u_lower_bounds - action > 0.0) or np.any(state - self.x_upper_bounds > 0.0) or np.any(self.x_lower_bounds - state > 0.0):
        if np.any(state - self.x_upper_bounds > 0.0) or np.any(self.x_lower_bounds - state > 0.0):
            self.violations += 1
        return
    
    def _zeroth_stage(self):

        if self.parameter_step_num < self.theta_p0.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_zeroth_stage()
        return observation, reward, termination, truncation, info
    
    def _simulate_zeroth_stage(self):

        
        # Extract current time
        t0 = self.time

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        # Extract and assign the recipe parameters
        theta_p0_num = self.theta_p0_func(self.theta_num)

        m_dot_f = float(theta_p0_num[0])
        T_R_set = float(theta_p0_num[1])
        K_P = float(theta_p0_num[2])
        K_I = float(theta_p0_num[3])
        m_tot_finished = float(theta_p0_num[4])
        

        # Define controllers
        outer_T_J = self.controllers_p0["outer_T_J"]
        outer_T_EK = self.controllers_p0["outer_T_EK"]
        
        outer_T_J.update_state_setpoint(T_R_set)
        outer_T_EK.update_state_setpoint(T_R_set)
        outer_T_J.update_action_setpoint(T_R_set)
        outer_T_EK.update_action_setpoint(T_R_set)

        setpoints = cd.DM([T_R_set, T_R_set])

        outer_T_J.update_K_P(K_P)
        outer_T_J.update_K_I(K_I)
        outer_T_EK.update_K_P(K_P)
        outer_T_EK.update_K_I(K_I)


        inner_T_J_in = self.controllers_p0["inner_T_J_in"]
        inner_T_EK_in = self.controllers_p0["inner_T_EK_in"]

        outer_T_J.reset()
        outer_T_EK.reset()
        inner_T_J_in.reset()
        inner_T_EK_in.reset()



        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0

        tol = 1e-1
        m_tot = float(state[0] + state[1] + state[2])

        # while self.time < t0 + 10/60 or (self.time < t0 + 0.5 and float(state[0] + state[1] + state[2]) < m_tot_finished and state[-2] < self.x_upper_bounds[-2]):
        while m_tot < m_tot_finished and state[-2] < self.x_upper_bounds[-2]:

            if state[2] >= (self.x_upper_bounds[-2] * self.w_AF + self.m_p_init - tol):
                if not global_truncation:
                    termination = True
                break
            
            # Define the control actions
            m_dot_f_num = np.clip(0.0 + m_dot_f * (self.time - t0), 0.0, 30e3)

            T_Jacket_outer_setpoint = outer_T_J.step(state[3])
            T_EK_outer_setpoint = outer_T_EK.step(state[3])

            sp = np.array([outer_T_J.state_sp, T_Jacket_outer_setpoint, T_EK_outer_setpoint]).reshape(-1,1)
            self.setpoint_data = np.vstack([self.setpoint_data, sp.T])

            inner_T_J_in.update_state_setpoint(T_Jacket_outer_setpoint)
            T_Jacket_in_num = inner_T_J_in.step(state[5])

            inner_T_EK_in.update_state_setpoint(T_EK_outer_setpoint)
            T_EK_in_num = inner_T_EK_in.step(state[6])

            old_setpoints = setpoints
            setpoints = cd.vertcat(T_Jacket_outer_setpoint, T_EK_outer_setpoint)
            delta_setpoints = setpoints - old_setpoints

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EK_in_num)

            

            state = self.x_num

            self._violation_check(state, action)

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)
            reward += self._reward_tracking(state, T_R_set, delta_setpoints)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            m_tot = float(self.x_num[0] + self.x_num[1] + self.x_num[2])

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True
        
        self.controllers_p0["outer_T_J"] = outer_T_J
        self.controllers_p0["outer_T_EK"] = outer_T_EK
        self.controllers_p0["inner_T_J_in"] = inner_T_J_in
        self.controllers_p0["inner_T_EK_in"] = inner_T_EK_in
        
        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = False
        info = {}
        return observation, reward, termination, global_truncation, info
    
    def _first_stage(self):

        if self.parameter_step_num < self.theta_p0.shape[0] + self.theta_p1.shape[0]:
        # if self.parameter_step_num < self.theta_p1.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_first_stage()

        return observation, reward, termination, truncation, info
    
    def _simulate_first_stage(self):

        # Extract current time
        t0 = self.time

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        # Extract and assign the recipe parameters
        theta_p1_num = self.theta_p1_func(self.theta_num)

        m_dot_f_slope = float(theta_p1_num[0])
        # m_dot_f_slope = 60 * 10 ** m_dot_f_slope
        # m_dot_f_yIntercept = float(theta_p1_num[1])
        # m_dot_f_yIntercept = 30 * 10 ** m_dot_f_yIntercept
        
        # NOTE: HERE!
        # T_R_set = float(theta_p1_num[2])
        # T_Jacket_set = float(theta_p1_num[3])
        # T_EK_set = float(theta_p1_num[4])

        # T_Jacket_in_set = float(theta_p1_num[5])
        # T_EHE_cool_in_num = float(theta_p1_num[6])

        # max_duration = float(theta_p1_num[7])

        T_R_set = float(theta_p1_num[1])
        K_P = float(theta_p1_num[2])
        K_I = float(theta_p1_num[3])
        # T_Jacket_set = float(theta_p1_num[2])
        # T_EK_set = float(theta_p1_num[3])

        # T_Jacket_in_set = float(theta_p1_num[4])
        # T_EHE_cool_in_num = float(theta_p1_num[5])

        # max_duration = float(theta_p1_num[6])

        # m_dot_f_max = float(theta_p1_num[7])

        max_duration = float(theta_p1_num[4])

        m_dot_f_max = float(theta_p1_num[5])

        # Define controllers
        # outer_K_P_max = 7
        # outer_K_I_max = 50
        outer_T_J = self.controllers_p1["outer_T_J"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_Jacket_set, K_P = outer_K_P_max, K_I = outer_K_I_max, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        outer_T_EK = self.controllers_p1["outer_T_EK"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_EK_set, K_P = outer_K_P_max, K_I = outer_K_I_max, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])
        # outer_T_J = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_Jacket_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        # outer_T_EK = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_EK_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        inner_T_J_in = self.controllers_p1["inner_T_J_in"] #Anti_Windup_Controller(state_sp = T_Jacket_set, action_sp = T_Jacket_in_set, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        inner_T_EK_in = self.controllers_p1["inner_T_EK_in"]# Anti_Windup_Controller(state_sp = T_EK_set, action_sp = T_EHE_cool_in_num, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])

        outer_T_J.reset()
        outer_T_EK.reset()
        inner_T_J_in.reset()
        inner_T_EK_in.reset()

        outer_T_J.update_state_setpoint(T_R_set)
        outer_T_EK.update_state_setpoint(T_R_set)
        # outer_T_J.update_action_setpoint(T_Jacket_set)
        # outer_T_EK.update_action_setpoint(T_EK_set)
        outer_T_J.update_action_setpoint(T_R_set)
        outer_T_EK.update_action_setpoint(T_R_set)

        setpoints = cd.DM([T_R_set, T_R_set])

        outer_T_J.update_K_P(K_P)
        outer_T_J.update_K_I(K_I)
        outer_T_EK.update_K_P(K_P)
        outer_T_EK.update_K_I(K_I)

        outer_T_J.inherit_integrated_error(self.controllers_p0["outer_T_J"], state[3])
        outer_T_EK.inherit_integrated_error(self.controllers_p0["outer_T_EK"], state[3])

        # inner_T_J_in.update_action_setpoint(T_Jacket_in_set)
        # inner_T_EK_in.update_action_setpoint(T_EHE_cool_in_num)
        inner_T_J_in.update_action_setpoint(273.15 + 90)
        inner_T_EK_in.update_action_setpoint(273.15 + 90)

        inner_T_J_in.inherit_integrated_error(self.controllers_p0["inner_T_J_in"], state[5])
        inner_T_EK_in.inherit_integrated_error(self.controllers_p0["inner_T_EK_in"], state[6])

        outer_T_J_base_K_P = outer_T_J.K_P
        outer_T_J_base_K_I = outer_T_J.K_I

        outer_T_EK_base_K_P = outer_T_EK.K_P
        outer_T_EK_base_K_I = outer_T_EK.K_I






        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0

        tol = 1e-1

        switched = False

        m_dot_f_base = self.u_num[0]
        while self.time < t0 + max_duration and state[-2] < self.x_upper_bounds[-2]:
            
            # Define the control actions
            # m_dot_f_num = np.clip(m_dot_f_yIntercept + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = self.u_upper_bounds[0])
            # m_dot_f_num = np.clip(0.0 + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = self.u_upper_bounds[0])
            m_dot_f_num = np.clip(m_dot_f_base + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = m_dot_f_max)
            
            if m_dot_f_num >= m_dot_f_max:
                break

            if state[2] >= (self.x_upper_bounds[-2] * self.w_AF + self.m_p_init - tol):
                if not global_truncation:
                    termination = True
                break
                

            m_tot = state[0] + state[1] + state[2]
            # progress_KP = 1  + 1 * (m_tot/self.m_tot_init - 1)
            progress_KP = 1
            # progress_KI = 1  + 0.75 * (m_tot/self.m_tot_init - 1)
            # progress_KI = 1 + 1.25 * m_dot_f_num / m_tot
            progress_KI = 1
            outer_T_J.update_K_P(outer_T_J_base_K_P * progress_KP)
            outer_T_EK.update_K_P(outer_T_EK_base_K_P * progress_KP)

            outer_T_J.update_K_I(outer_T_J_base_K_I * progress_KI)
            outer_T_EK.update_K_I(outer_T_EK_base_K_I * progress_KI)


            T_Jacket_outer_setpoint = outer_T_J.step(state[3])
            T_EK_outer_setpoint = outer_T_EK.step(state[3])

            sp = np.array([T_R_set, T_Jacket_outer_setpoint, T_EK_outer_setpoint]).reshape(-1,1)
            self.setpoint_data = np.vstack([self.setpoint_data, sp.T])

            inner_T_J_in.update_state_setpoint(T_Jacket_outer_setpoint)
            inner_T_J_in.update_action_setpoint(T_Jacket_outer_setpoint)
            T_Jacket_in_num = inner_T_J_in.step(state[5])

            inner_T_EK_in.update_state_setpoint(T_EK_outer_setpoint)
            inner_T_EK_in.update_action_setpoint(T_EK_outer_setpoint)
            T_EK_in_num = inner_T_EK_in.step(state[6])

            old_setpoints = setpoints
            setpoints = cd.vertcat(T_Jacket_outer_setpoint, T_EK_outer_setpoint)
            delta_setpoints = setpoints - old_setpoints

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EK_in_num)

            state = self.x_num

            self._violation_check(state, action)

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)
            reward += self._reward_tracking(state, T_R_set, delta_setpoints)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True

        
        self.controllers_p1["outer_T_J"] = outer_T_J
        self.controllers_p1["outer_T_EK"] = outer_T_EK
        self.controllers_p1["inner_T_J_in"] = inner_T_J_in
        self.controllers_p1["inner_T_EK_in"] = inner_T_EK_in

        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = False
        info = {}

        return observation, reward, termination, global_truncation, info

    def _second_stage(self):

        if self.parameter_step_num < self.theta_p0.shape[0] + self.theta_p1.shape[0] + self.theta_p2.shape[0]:
        # if self.parameter_step_num < self.theta_p1.shape[0] + self.theta_p2.shape[0]:
            observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
            reward = 0
            termination = False
            truncation = False
            info = {}
        else:
            observation, reward, termination, truncation, info = self._simulate_second_stage()

        return observation, reward, termination, truncation, info
    
    def _simulate_second_stage(self):
        
        # Extract current time
        t0 = self.time

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        # Extract and assign the recipe parameters
        theta_p2_num = self.theta_p2_func(self.theta_num)

        # m_dot_f_slope = float(theta_p2_num[0])
        # m_dot_f_yIntercept = float(theta_p2_num[1])
        
        # # NOTE: HERE!
        # T_R_set = float(theta_p2_num[2])
        # T_Jacket_set = float(theta_p2_num[3])
        # T_EK_set = float(theta_p2_num[4])

        # T_Jacket_in_set = float(theta_p2_num[5])
        # T_EHE_cool_in_num = float(theta_p2_num[6])

        # max_duration = float(theta_p2_num[7])
        
        # m_dot_f_slope = float(theta_p2_num[0])
        # m_dot_f_yIntercept = float(theta_p2_num[1])

        m_dot_f_slope = 0.0
        m_dot_f_yIntercept = self.u_num[0]
        
        # NOTE: HERE!
        T_R_set = float(theta_p2_num[0])
        K_P = float(theta_p2_num[1])
        K_I = float(theta_p2_num[2])
        # T_Jacket_set = float(theta_p2_num[3])
        # T_EK_set = float(theta_p2_num[4])

        # T_Jacket_in_set = float(theta_p2_num[4])
        # T_EHE_cool_in_num = float(theta_p2_num[5])


        # Define controller variables
        outer_T_J = self.controllers_p2["outer_T_J"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_Jacket_set, K_P = 5, K_I = 75, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        outer_T_EK = self.controllers_p2["outer_T_EK"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_EK_set, K_P = 3, K_I = 75, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])
        # outer_T_J = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_Jacket_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        # outer_T_EK = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_EK_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        inner_T_J_in = self.controllers_p2["inner_T_J_in"] #Anti_Windup_Controller(state_sp = T_Jacket_set, action_sp = T_Jacket_in_set, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        inner_T_EK_in = self.controllers_p2["inner_T_EK_in"] #Anti_Windup_Controller(state_sp = T_EK_set, action_sp = T_EHE_cool_in_num, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])

        outer_T_J.reset()
        outer_T_EK.reset()
        inner_T_J_in.reset()
        inner_T_EK_in.reset()

        outer_T_J.update_K_P(K_P)
        outer_T_J.update_K_I(K_I)
        outer_T_EK.update_K_P(K_P)
        outer_T_EK.update_K_I(K_I)

        outer_T_J.update_state_setpoint(T_R_set)
        outer_T_EK.update_state_setpoint(T_R_set)
        # outer_T_J.update_action_setpoint(T_Jacket_set)
        # outer_T_EK.update_action_setpoint(T_EK_set)
        outer_T_J.update_action_setpoint(T_R_set)
        outer_T_EK.update_action_setpoint(T_R_set)

        setpoints = cd.DM([T_R_set, T_R_set])

        outer_T_J.inherit_integrated_error(self.controllers_p1["outer_T_J"], state[3])
        outer_T_EK.inherit_integrated_error(self.controllers_p1["outer_T_EK"], state[3])

        set_1 = outer_T_J.step(state[3])
        set_2 = outer_T_EK.step(state[3])
        # inner_T_J_in.update_action_setpoint(T_Jacket_in_set)
        # inner_T_EK_in.update_action_setpoint(T_EHE_cool_in_num)
        inner_T_J_in.update_action_setpoint(set_1)
        inner_T_EK_in.update_action_setpoint(set_2)

        inner_T_J_in.inherit_integrated_error(self.controllers_p1["inner_T_J_in"], state[5])
        inner_T_EK_in.inherit_integrated_error(self.controllers_p1["inner_T_EK_in"], state[6])



        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0
        
        tol = 1e-1
        no_feeding = False
        controler_switched = False
        # while self.time < t0 + max_duration and state[-2] <= self.x_upper_bounds[-2]:
        m_dot_f_num = 100
        while state[2] <= (self.x_upper_bounds[-2] * self.w_AF + self.m_p_init - tol) and (m_dot_f_num > tol or not state[-2] <= self.x_upper_bounds[-2]):
            
            # Define the control actions
            m_dot_f_num = np.clip(m_dot_f_yIntercept + m_dot_f_slope * (self.time - t0), a_min = self.u_lower_bounds[0], a_max = self.u_upper_bounds[0])

            if state[-2] >= self.x_upper_bounds[-2]:
                m_dot_f_num = 0
                state[-2] = self.x_num[-2] =  np.clip(state[-2], a_min = self.x_lower_bounds[-2], a_max = self.x_upper_bounds[-2])

                no_feeding = True

            if no_feeding and not controler_switched:
                outer_T_J_old = outer_T_J
                outer_T_EK_old = outer_T_EK
                # inner_T_J_in_old = inner_T_J_in
                # inner_T_EK_in_old = inner_T_EK_in

                outer_T_J = self.controllers_p3["outer_T_J"]
                outer_T_EK = self.controllers_p3["outer_T_EK"]
                # inner_T_J_in = self.controllers_p3["inner_T_J_in"]
                # inner_T_EK_in = self.controllers_p3["inner_T_EK_in"]

                outer_T_J.reset()
                outer_T_EK.reset()
                inner_T_J_in.reset()

                outer_T_J.update_state_setpoint(T_R_set)
                outer_T_EK.update_state_setpoint(T_R_set)
                # outer_T_J.update_action_setpoint(T_Jacket_set)
                # outer_T_EK.update_action_setpoint(T_EK_set)
                outer_T_J.update_action_setpoint(T_R_set)
                outer_T_EK.update_action_setpoint(T_R_set)

                outer_T_J.inherit_integrated_error(outer_T_J_old, T_R_set)
                outer_T_EK.inherit_integrated_error(outer_T_EK_old, T_R_set)

                # inner_T_J_in.update_action_setpoint(273.15 + 90)
                # inner_T_EK_in.update_action_setpoint(273.15 + 90)

                # inner_T_J_in.inherit_integrated_error(inner_T_J_in_old, state[5])
                # inner_T_EK_in.inherit_integrated_error(inner_T_EK_in_old, state[6])

                controler_switched = True


            
            T_Jacket_outer_setpoint = outer_T_J.step(state[3])
            T_EK_outer_setpoint = outer_T_EK.step(state[3])

            sp = np.array([T_R_set, T_Jacket_outer_setpoint, T_EK_outer_setpoint]).reshape(-1,1)
            self.setpoint_data = np.vstack([self.setpoint_data, sp.T])

            inner_T_J_in.update_state_setpoint(T_Jacket_outer_setpoint)
            inner_T_J_in.update_action_setpoint(T_Jacket_outer_setpoint)
            T_Jacket_in_num = inner_T_J_in.step(state[5])

            inner_T_EK_in.update_state_setpoint(T_EK_outer_setpoint)
            inner_T_EK_in.update_action_setpoint(T_EK_outer_setpoint)
            T_EK_in_num = inner_T_EK_in.step(state[6])

            old_setpoints = setpoints
            setpoints = cd.vertcat(T_Jacket_outer_setpoint, T_EK_outer_setpoint)
            delta_setpoints = setpoints - old_setpoints

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EK_in_num)

            state = self.x_num

            self._violation_check(state, action)

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)
            reward += self._reward_tracking(state, T_R_set, delta_setpoints)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True

        self.controllers_p2["outer_T_J"] = outer_T_J
        self.controllers_p2["outer_T_EK"] = outer_T_EK
        self.controllers_p2["inner_T_J_in"] = inner_T_J_in
        self.controllers_p2["inner_T_EK_in"] = inner_T_EK_in

        state[-2] = self.x_num[-2] =  np.clip(state[-2], a_min = self.x_lower_bounds[-2], a_max = self.x_upper_bounds[-2])
        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = True
        info = {}

        return observation, reward, termination, global_truncation, info

    def _simulate_third_stage(self):
        # Extract current time
        t0 = self.time
        self.max_time = self.time + 1.5

        # Extract and assign the recipe parameters
        theta_p3_num = self.theta_p3_func(self.theta_num)

        
        # NOTE: HERE!
        T_R_set = float(theta_p3_num[0])
        T_Jacket_set = float(theta_p3_num[1])
        T_EK_set = float(theta_p3_num[2])

        T_Jacket_in_set = float(theta_p3_num[3])
        T_EHE_cool_in_num = float(theta_p3_num[4])


        # Define controller variables
        outer_T_J = self.controllers_p3["outer_T_J"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_Jacket_set, K_P = 15, K_I = 125, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        outer_T_EK = self.controllers_p3["outer_T_EK"] #Anti_Windup_Controller(state_sp = T_R_set, action_sp = T_EK_set, K_P = 15, K_I = 125, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])
        # outer_T_J = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_Jacket_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[5], action_upper = self.x_upper_bounds[5])
        # outer_T_EK = Anti_Windup_Controller(state_sp = 273.15 + 90, action_sp = T_EK_set, K_P = 50, K_I = 0.01, dt = self.integration_settings.dt, action_lower = self.x_lower_bounds[6], action_upper = self.x_upper_bounds[6])

        inner_T_J_in = self.controllers_p3["inner_T_J_in"] #Anti_Windup_Controller(state_sp = T_Jacket_set, action_sp = T_Jacket_in_set, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[1], action_upper = self.u_upper_bounds[1])
        inner_T_EK_in = self.controllers_p3["inner_T_EK_in"] #Anti_Windup_Controller(state_sp = T_EK_set, action_sp = T_EHE_cool_in_num, K_P = 1.1, K_I = 50, dt = self.integration_settings.dt, action_lower = self.u_lower_bounds[2], action_upper = self.u_upper_bounds[2])

        outer_T_J.reset()
        outer_T_EK.reset()
        inner_T_J_in.reset()
        inner_T_EK_in.reset()

        outer_T_J.update_state_setpoint(T_R_set)
        outer_T_EK.update_state_setpoint(T_R_set)
        outer_T_J.update_action_setpoint(T_Jacket_set)
        outer_T_EK.update_action_setpoint(T_EK_set)

        outer_T_J.inherit_integrated_error(self.controllers_p2["outer_T_J"])
        outer_T_EK.inherit_integrated_error(self.controllers_p2["outer_T_EK"])

        inner_T_J_in.update_action_setpoint(T_Jacket_in_set)
        inner_T_EK_in.update_action_setpoint(T_EHE_cool_in_num)

        inner_T_J_in.inherit_integrated_error(self.controllers_p2["inner_T_J_in"])
        inner_T_EK_in.inherit_integrated_error(self.controllers_p2["inner_T_EK_in"])

        # Extract the first state and action for initialization
        state = self.x_num
        action = self.u_num

        global_truncation = False
        truncation = self._truncated_check(state, action)
        if truncation:
            global_truncation = True

        # Initialize the reward
        reward = 0
        
        while self.time < self.max_time:
            
            # Define the control actions
            m_dot_f_num = 0.0e3
            
            T_Jacket_outer_setpoint = outer_T_J.step(state[3])
            T_EK_outer_setpoint = outer_T_EK.step(state[3])

            sp = np.array([T_R_set, T_Jacket_outer_setpoint, T_EK_outer_setpoint]).reshape(-1,1)
            self.setpoint_data = np.vstack([self.setpoint_data, sp.T])

            inner_T_J_in.update_state_setpoint(T_Jacket_outer_setpoint)
            T_Jacket_in_num = inner_T_J_in.step(state[5])

            inner_T_EK_in.update_state_setpoint(T_EK_outer_setpoint)
            T_EK_in_num = inner_T_EK_in.step(state[6])

            action = cd.vertcat(m_dot_f_num, T_Jacket_in_num, T_EK_in_num)

            state = self.x_num

            self.x_data = np.vstack([self.x_data, state.T])
            self.u_data = np.vstack([self.u_data, action.full().T])

            reward += self._compute_stage_cost(state, action)

            self.x_num = self.integrator(x0 = state, p = action)["xf"]
            self.u_num = action

            self.time += self.integration_settings.dt

            truncation = self._truncated_check(state, action)
            if truncation:
                global_truncation = True

        observation = self.s_num = self.s_func(self.x_num, self.theta_num, self.parameter_step_num)
        termination = True
        info = {}

        return observation, reward, termination, global_truncation, info

    
    def reset(self, seed: int = None, options = None):

        observation, info = super().reset(seed = seed, options = options)

        self.setpoint_data = np.empty((0, self.setpoint_data.shape[1]))

        self.m_p_init = self.x_num[1]
        self.m_tot_init = self.x_num[0] + self.x_num[1] + self.x_num[2]

        self.violations = 0

        return observation, info

class Poly_reactor_SB3_time(Poly_reactor_SB3_cascade):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_time = 10
        return
    
    def _setup_stage_cost(self):
        
        self.relaxation_weights_x = cd.DM.ones(self.x.shape) * 1e5
        self.relaxation_weights_u = cd.DM.ones(self.u.shape) * 1e5

        # Maximize the product of the reactor 
        # self.stage_cost = 20e3 - self.x[2] # Maybe we have to look at this again
        self.stage_cost = 0

        # Penalize constraint violations
        self.stage_cost_penalties = cd.fmax(self.x_lower_bounds - self.x, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.x - self.x_upper_bounds, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.u_lower_bounds - self.u, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u
        self.stage_cost_penalties += cd.fmax(self.u - self.u_upper_bounds, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u

        self.stage_cost += self.stage_cost_penalties

        # Scaling
        self.stage_cost *= 1 / 20e3 
        self.stage_cost_penalties *= 1 / 20e3 

        # self.stage_cost = cd.log10(self.stage_cost)
        # self.stage_cost_penalties = cd.log10(self.stage_cost_penalties)

        # Adapting to step length
        self.stage_cost *= self.integration_settings.dt
        self.stage_cost_penalties *= self.integration_settings.dt

        # Add the current time
        self.stage_cost += self.integration_settings.dt


        self._stage_cost_func = cd.Function("stage_cost", [self.x, self.u], [self.stage_cost], ["x", "u",], ["stage_cost"])
        return

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.time >= self.max_time:
            terminated = False
            truncated = True
        return observation, reward, terminated, truncated, info

class Poly_reactor_SB3_hybrid(Poly_reactor_SB3_time):

    def _setup_stage_cost(self):
        
        self.relaxation_weights_x = cd.DM.ones(self.x.shape) * 1e5
        self.relaxation_weights_u = cd.DM.ones(self.u.shape) * 1e5

        # Maximize the product of the reactor 
        self.stage_cost = 20e3 - self.x[2] # Maybe we have to look at this again
        # self.stage_cost = 0

        # Penalize constraint violations
        self.stage_cost_penalties = cd.fmax(self.x_lower_bounds - self.x, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.x - self.x_upper_bounds, cd.DM.zeros(self.x.shape)).T @ self.relaxation_weights_x
        self.stage_cost_penalties += cd.fmax(self.u_lower_bounds - self.u, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u
        self.stage_cost_penalties += cd.fmax(self.u - self.u_upper_bounds, cd.DM.zeros(self.u.shape)).T @ self.relaxation_weights_u

        self.stage_cost += self.stage_cost_penalties


        # Scaling
        self.stage_cost *= 1 / 20e3 
        self.stage_cost_penalties *= 1 / 20e3 

        # self.stage_cost = cd.log10(self.stage_cost)
        # self.stage_cost_penalties = cd.log10(self.stage_cost_penalties)

        # Adapting to step length
        self.stage_cost *= self.integration_settings.dt
        self.stage_cost_penalties *= self.integration_settings.dt

        # Add the current time
        self.stage_cost += self.integration_settings.dt


        self._stage_cost_func = cd.Function("stage_cost", [self.x, self.u], [self.stage_cost], ["x", "u",], ["stage_cost"])
        return

class Poly_reactor_MPC(Poly_reactor_SB3_cascade):

    def step(self, action: np.ndarray):
        
        self.x_data = np.vstack([self.x_data, self.x_num.T])
        self.u_data = np.vstack([self.u_data, action.T])

        x_next = self.integrator(x0 = self.x_num, p = action)["xf"]

        self.time += self.integration_settings.dt

        reward = self._compute_stage_cost(self.x_num, action)

        terminated = False
        truncated = self._truncated_check(self.x_num, action)
        
        info = {}
        
        self.x_num = x_next.full().copy()

        return x_next.full(), reward, terminated, truncated, info
    
    def reset(self, seed: int = None, options = None):
        state, info = super().reset(seed = seed, options = options)
        state = self.observation_unscaling_func(state)
        x, theta, step = self.rev_s(state)
        return x.full(), info


class Poly_reactor_SB3_direct_time(gym.Env):

    def __init__(self, seed: int = 1234):

        # Initialize the flags
        self.flags = flags()

        self.integration_settings = integration_settings()

        
        # Setup the system equations
        x, u = self._setup_system()
        observation = self.observation = cd.vertcat(x, u)
        action = self.action = u

        self.observation_func = cd.Function("observation_func", [x, u], [observation], ["x", "u"], ["observation"])
        self.inverse_observation_func = cd.Function("inverse_observation_func", [observation], [x, u], ["observation"], ["x", "u"])



        # Setup the scaling
        self._setup_bounds()
        self._setup_bounds_for_initialization()

        self.x_scaling_func = cd.Function("x_scaling", [x], [(x - self.x_lower_bounds) / (self.x_upper_bounds  - self.x_lower_bounds)], ["x"], ["x_scaled"])
        self.x_unscaling_func = cd.Function("x_unscaling", [x], [self.x_lower_bounds + x * (self.x_upper_bounds - self.x_lower_bounds)], ["x_scaled"], ["x"])
        self.u_scaling_func = cd.Function("u_scaling", [u], [(u - self.u_lower_bounds) / (self.u_upper_bounds  - self.u_lower_bounds)], ["u"], ["u_scaled"])
        self.u_unscaling_func = cd.Function("u_unscaling", [u], [self.u_lower_bounds + u * (self.u_upper_bounds - self.u_lower_bounds)], ["u_scaled"], ["u"])
        self.observation_scaling_func = cd.Function("observation_scaling", [observation], [(observation - self.observation_lower_bounds) / (self.observation_upper_bounds  - self.observation_lower_bounds)], ["observation"], ["observation_scaled"])
        self.observation_unscaling_func = cd.Function("observation_unscaling", [observation], [self.observation_lower_bounds + observation * (self.observation_upper_bounds - self.observation_lower_bounds)], ["observation_scaled"], ["observation"])
        

        # Setup spaces
        self.action_space = spaces.Box(low = np.zeros(action.shape), high = np.ones(action.shape), dtype = np.float32)
        self.observation_space = spaces.Box(low = np.zeros(observation.shape) - np.inf, high = np.ones(observation.shape) + np.inf, dtype = np.float32)

        # Setup the stage cost
        # self._setup_stage_cost()

        self.termination_list = deque(maxlen = 50)
        self.truncation_list = deque(maxlen = 50)
        self.batch_time_list = deque(maxlen = 50)
        self.violations_list = deque(maxlen = 50)
        self.violations: int = 0

        # Initialize the seed to get reproducible results
        self.seed: int = seed
        self.rng = np.random.default_rng(seed = seed)

        # Initialize the time
        self.time: float = 0.0      # NOTE: In hours
        self.max_time: float = 3.0  # NOTE: In hours
        
        # Initialize the data structure
        self.observation_data = np.empty((0, observation.shape[0]))
        self.x_data = np.empty((0, x.shape[0]))
        self.u_data = np.empty((0, u.shape[0]))
        return


    def _setup_system(self):

        # System equations
        x, u = self._setup_system_equations()

        # idas
        self._setup_idas()

        return x, u
    
    def _setup_system_equations(self):
        # Certain parameters
        R           = 8.314    			#gas constant
        T_F         = 25 + 273.15       #feed temperature
        E_a         = 8500.0     			#activation energy
        delH_R      = 950.0*1.00      			#sp reaction enthalpy
        A_tank      = 65.0       			#area heat exchanger surface jacket 65

        k_0         = 7.0*1.00      	#sp reaction rate
        k_U2        = 32.0     	#reaction parameter 1
        k_U1        = 4.0      	#reaction parameter 2
        w_WF        = .333      #mass fraction water in feed
        self.w_AF = w_AF       = .667      #mass fraction of A in feed

        m_M_KW      = 5000.0      #mass of coolant in jacket
        fm_M_KW     = 300000.0    #coolant flow in jacket 300000;
        m_AWT_KW    = 1000.0      #mass of coolant in EHE
        fm_AWT_KW   = 100000.0    #coolant flow in EHE
        m_AWT       = 200.0       #mass of product in EHE
        fm_AWT      = 20000.0     #product flow in EHE
        m_S         = 39000.0     #mass of reactor steel

        c_pW        = 4.2      #sp heat cap coolant
        c_pS        = .47       #sp heat cap steel
        c_pF        = 3.0         #sp heat cap feed
        self.c_pR        = 5.0         #sp heat cap reactor contents

        k_WS        = 17280.0     #heat transfer coeff water-steel
        k_AS        = 3600.0      #heat transfer coeff monomer-steel
        k_PS        = 360.0       #heat transfer coeff product-steel

        alfa        = 5*20e4*3.6

        p_1         = 1.0

        self.delH_R = 950.0
        k_0 =   7.0

        # States:
        m_W = cd.SX.sym('m_W')                  # NOTE: In kg
        m_A = cd.SX.sym('m_A')                  # NOTE: In kg
        m_P = cd.SX.sym('m_P')                  # NOTE: In kg

        T_R = cd.SX.sym('T_R')                  # NOTE: In K
        T_S = cd.SX.sym('T_S')                  # NOTE: In K
        Tout_M = cd.SX.sym('Tout_M')            # NOTE: In K

        T_EK = cd.SX.sym('T_EK')                # NOTE: In K
        Tout_AWT = cd.SX.sym('Tout_AWT')        # NOTE: In K

        accum_monom = cd.SX.sym('accum_monom')  # NOTE: In kg
        T_adiab = cd.SX.sym('T_adiab')          # NOTE: In K

        self.x = cd.vertcat(m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_monom, T_adiab)
        # self.x_next = cd.SX.sym("x_next", self.x.shape)

        self.x_scaling = cd.DM([10e3, 10e3, 10e3, 1e2, 1e2, 1e2, 1e2, 1e2, 10e3, 1e2])

        # Action:
        m_dot_f = cd.SX.sym('m_dot_f')
        T_in_M =  cd.SX.sym('T_in_M')
        T_in_EK = cd.SX.sym('T_in_EK')

        self.u = cd.vertcat(m_dot_f, T_in_M, T_in_EK)

        # algebraic equations (Just helping expressions)
        U_m    = m_P / (m_A + m_P)
        m_ges  = m_W + m_A + m_P
        k_R1   = k_0 * cd.exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_R2   = k_0 * cd.exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)
        m_A_R  = m_A * (1 - m_AWT / m_ges)


        # Differential equations
        dot_m_W = m_dot_f * w_WF
        dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
        dot_m_P = k_R1 * m_A_R + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)

        dot_T_R = 1./(self.c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * self.c_pR * (T_R - T_EK)) + (self.delH_R * k_R1 * m_A_R))
        dot_T_S =  1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)))
        dot_Tout_M = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)))
        
        dot_T_EK = 1./(self.c_pR * m_AWT)   * ((fm_AWT * self.c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * self.delH_R))
        dot_Tout_AWT = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))
        
        dot_accum_monom = m_dot_f
        dot_T_adiab = self.delH_R/(m_ges*self.c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*self.delH_R/(m_ges*m_ges*self.c_pR))+dot_T_R

        self.rhs = cd.vertcat(dot_m_W, dot_m_A, dot_m_P, dot_T_R, dot_T_S, dot_Tout_M, dot_T_EK, dot_Tout_AWT, dot_accum_monom, dot_T_adiab)
        

        self.flags.system_set = True
        return self.x, self.u
    
    def _setup_idas(self):

        self.dae_dict = {
            "x": self.x,
            "p": self.u,
            "ode": self.rhs,
        }

        if self.integration_settings.opts is None:
            self.integration_settings.opts = {
                # "rf": "kinsol",
                # "rootfinder_options": {"strategy": "linesearch"}
                # "verbose": True,
            }

        # Build the integrator
        self.integrator = cd.integrator("system", "idas", self.dae_dict, 0.0, self.integration_settings.dt, self.integration_settings.opts)
        
        # Simplify the integration in- and outputs
        integrator_inputs = self.integrator.name_in()[:3]
        intgrator_outputs = self.integrator.name_out()[:2]
        self.integrator = self.integrator.factory("system", integrator_inputs, intgrator_outputs)

        self.flags.integrator_set = True
        return
    
    def _setup_bounds(self):     

        # Setup the scaling
        self.x_lower_bounds = np.array([
            0.0e3,
            0.0e3,
            0.0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            273.15 + 60,
            0.0e3,
            273.15 + 90 - 2.0
            ]).reshape(-1,1)
        
        self.x_upper_bounds = np.array([
            30e3,
            30e3,
            30e3,
            273.15 + 90.0 + 2.0,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            100. + 273.15,
            30e3,
            109. + 273.15,
            ]).reshape(-1,1)


        # As we clip everything this should not be of any issue
        self.u_lower_bounds = np.array([0., 273.15 + 60., 273.15 + 60.]).reshape(-1,1)
        self.u_upper_bounds = np.array([30e3, 273.15 + 100., 273.15 + 100.]).reshape(-1,1)


        self.observation_lower_bounds = self.observation_func(self.x_lower_bounds, self.u_lower_bounds)
        self.observation_upper_bounds = self.observation_func(self.x_upper_bounds, self.u_upper_bounds)
    
    def _setup_bounds_for_initialization(self):     

        # Setup the scaling
        self.x_lower_bounds_for_init = np.array([
            10.0e3,
            0.0e3,
            0.0e3,
            273.15 + 90.0 - 2.0,
            273.15 + 90.0 - 5.0,
            273.15 + 90.0 - 5.0,
            273.15 + 90.0 - 5.0,
            273.15 + 90.0 - 5.0,
            0.0e3,
            273.15 + 90 - 2.0
            ]).reshape(-1,1)
        
        self.x_upper_bounds_for_init = np.array([
            15.0e3,
            5.0e3,
            0.0e3,
            273.15 + 90.0 + 2.0,
            273.15 + 90.0 + 5.0,
            273.15 + 90.0 + 5.0,
            273.15 + 90.0 + 5.0,
            273.15 + 90.0 + 5.0,
            0.0e3,
            100. + 273.15,
            ]).reshape(-1,1)
        
        self.u_lower_bounds_for_init = np.array([0.0e3, 273.15 + 60., 273.15 + 60.]).reshape(-1,1)
        self.u_upper_bounds_for_init = np.array([0.0e3, 273.15 + 100., 273.15 + 100.]).reshape(-1,1)
    
    def _violation_check(self, state: np.ndarray, action: np.array):
        if np.any(action - self.u_upper_bounds > 0.0) or np.any(self.u_lower_bounds - action > 0.0) or np.any(state - self.x_upper_bounds > 0.0) or np.any(self.x_lower_bounds - state > 0.0):
            self.violations += 1
        return
    
    def _truncated_check(self, state: np.ndarray, action: np.ndarray) -> bool:
        truncated = False

        if self.time >= self.max_time:
            truncated = True
            return truncated

        action_tolerance = 0.1 # Percentage of range from min to max
        delta_a_max = (self.u_upper_bounds - self.u_lower_bounds) * action_tolerance

        if np.any(action - self.u_upper_bounds > delta_a_max) or np.any(self.u_lower_bounds - action > delta_a_max):
            truncated = True
            return truncated

        state_tolerance = 0.1 # Percentage of range from min to max
        delta_s_max = (self.x_upper_bounds - self.x_lower_bounds) * state_tolerance
        delta_s_max[3] = 4.0

        if np.any(state - self.x_upper_bounds > delta_s_max) or np.any(self.x_lower_bounds - state > delta_s_max):
            truncated = True
            return truncated

        return truncated
    
    def _terminated_check(self, state: np.ndarray, action: np.ndarray) -> bool:
        terminated = False

        if state[2] >= self.max_product:
            terminated = True
            return terminated
        
        return terminated
    
    def step(self, action: np.ndarray):
        
        # Rescale the action
        unscaled_action = self.u_unscaling_func(action).full()
        
        # Extract the current state and action
        state, previous_action = self.inverse_observation_func(self.observation_num)
        
        # Calculate the reward
        reward = self.calc_reward(state, previous_action, unscaled_action)

        # Integrate the system
        next_state = self.integrator(x0 = state, p = unscaled_action)["xf"].full()

        # Update the time
        self.time += self.integration_settings.dt

        # Extract the next state
        observation = self.observation_num = self.observation_func(next_state, unscaled_action)
        self.x_num = next_state.copy()
        self.u_num = unscaled_action.copy()

        self._violation_check(self.x_num, self.u_num)

        truncated = self._truncated_check(self.x_num, self.u_num)
        if not truncated:
            terminated = self._terminated_check(self.x_num, self.u_num)
        else:
            terminated = False

        if truncated or terminated:
            self.termination_list[-1] = terminated
            self.truncation_list[-1] = truncated
            self.batch_time_list[-1] = self.time
            self.violations_list[-1] = self.violations


        # Make sure that the agent just sees the scaled observation
        observation = self.observation_scaling_func(observation)

        # Dummy information
        info = {}
        self.current_step += 1

        # Update the stored data
        self.observation_data = np.vstack([self.observation_data, self.observation_num.T])
        self.x_data = np.vstack([self.x_data, self.x_num.T])
        self.u_data = np.vstack([self.u_data, self.u_num.T])

        return observation, reward, terminated, truncated, info
    
    def calc_reward(self, state: np.ndarray, old_action: np.ndarray, action: np.ndarray):
        reward = 0.0

        # Penalize large steps in the actions
        delta_action = action - old_action
        R_mat = np.diag([0.002, 0.004, 0.002])
        reward += float(delta_action.T @ R_mat @ delta_action)
        
        # Penalize constraint violations
        penalty = 0.0

        w_x = np.ones(self.x.shape) * 1e5
        penalty += float(w_x.T @ np.clip(state - self.x_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_x.T @ np.clip(self.x_lower_bounds - state, a_min = 0.0, a_max = np.inf))

        w_u = np.ones(self.u.shape) * 1e5
        penalty += float(w_u.T @ np.clip(action - self.u_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_u.T @ np.clip(self.u_lower_bounds - action, a_min = 0.0, a_max = np.inf))

        reward += penalty

        # Maximize the mass of product
        # reward += float(20e3 - state[2])

        # Scale the reward
        reward *= 1 / 20e3

        # Adapt according to time step
        reward *= self.integration_settings.dt

        # Add the current time step to make the batch as short as possible
        reward += self.integration_settings.dt # NOTE: Maybe add later

        # Inverse signs to make it a reward and not a stage cost
        reward *= -1

        # Scale the reward
        reward *= 10
        return reward

    def reset(self, seed: int = None, options = None):
        
        nx = self.x.shape[0]
        nu = self.u.shape[0]

        # Sample uniform in the initialization space
        x_num = self.x_lower_bounds_for_init + (self.x_upper_bounds_for_init - self.x_lower_bounds_for_init) * self.rng.uniform(low = 0.0, high = 1.0, size = (nx, 1))
        
        # Set some constraints on the initial state
        x_num[4] = x_num[3].copy() # Steel temp = reactor temp

        # Calculating a feasible T_adiab
        x_num[9] = x_num[3] + self.delH_R / self.c_pR * x_num[1] / (x_num[0] + x_num[1] + x_num[2])

        if x_num[9] > self.x_upper_bounds_for_init[-1]:
            x_num[9] = self.x_upper_bounds_for_init[-1]
            phi = (x_num[9] - x_num[3]) * self.c_pR / self.delH_R
            x_num[1] = phi / (1 - phi) * (x_num[0] + x_num[2])


        # Save the computed initial state
        self.x_num = x_num.copy() 

        # u_num = self.u_lower_bounds_for_init + (self.u_upper_bounds_for_init - self.u_lower_bounds_for_init) * self.rng.uniform(low = 0.0, high = 1.0, size = (nu, 1))
        u_num = np.array([0, 273.15 + 60, 273.15 + 60]).reshape(-1,1)

        self.u_num = u_num.copy()

        # Calculate max possible product
        tol = 1e0
        self.max_product = self.x_num[2] + self.x_num[1] + self.x_upper_bounds[-2] * self.w_AF - tol

        # Stack everything to the whole observation
        observation = self.observation_num = self.observation_func(self.x_num, self.u_num).full()

        # Make sure that the agent just sees the scaled version
        observation = self.observation_scaling_func(observation)

        # Reset the time
        self.time = 0.0

        # Reset the data because it will blow up otherwise
        self.observation_data = self.observation_num.T.copy()
        self.x_data = self.x_num.T
        self.u_data = self.u_num.T

        # Say that everythin is initialized now
        self.flags.ic_set = True
        self.current_step = 0

        self.termination_list.append(False)
        self.truncation_list.append(False)
        self.batch_time_list.append(0.0)
        self.violations = 0
        self.violations_list.append(self.violations)

        return observation, {}

class Poly_reactor_SB3_direct_product(Poly_reactor_SB3_direct_time):

    def calc_reward(self, state: np.ndarray, old_action: np.ndarray, action: np.ndarray):
        reward = 0.0

        # Penalize large steps in the actions
        delta_action = action - old_action
        R_mat = np.diag([0.002, 0.004, 0.002])
        reward += float(delta_action.T @ R_mat @ delta_action)
        
        # Penalize constraint violations
        penalty = 0.0

        w_x = np.ones(self.x.shape) * 1e5
        penalty += float(w_x.T @ np.clip(state - self.x_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_x.T @ np.clip(self.x_lower_bounds - state, a_min = 0.0, a_max = np.inf))

        w_u = np.ones(self.u.shape) * 1e5
        penalty += float(w_u.T @ np.clip(action - self.u_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_u.T @ np.clip(self.u_lower_bounds - action, a_min = 0.0, a_max = np.inf))

        reward += penalty

        # Maximize the mass of product
        reward += float(20e3 - state[2])

        # Scale the reward
        reward *= 1 / 20e3

        # Adapt according to time step
        reward *= self.integration_settings.dt

        # Add the current time step to make the batch as short as possible
        # reward += self.integration_settings.dt # NOTE: Maybe add later

        # Inverse signs to make it a reward and not a stage cost
        reward *= -1

        # Scale the reward
        reward *= 10
        return reward
    
class Poly_reactor_SB3_direct_hybrid(Poly_reactor_SB3_direct_time):

    def calc_reward(self, state: np.ndarray, old_action: np.ndarray, action: np.ndarray):
        reward = 0.0

        # Penalize large steps in the actions
        delta_action = action - old_action
        R_mat = np.diag([0.002, 0.004, 0.002])
        reward += float(delta_action.T @ R_mat @ delta_action)
        
        # Penalize constraint violations
        penalty = 0.0

        w_x = np.ones(self.x.shape) * 1e5
        penalty += float(w_x.T @ np.clip(state - self.x_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_x.T @ np.clip(self.x_lower_bounds - state, a_min = 0.0, a_max = np.inf))

        w_u = np.ones(self.u.shape) * 1e5
        penalty += float(w_u.T @ np.clip(action - self.u_upper_bounds, a_min = 0.0, a_max = np.inf))
        penalty += float(w_u.T @ np.clip(self.u_lower_bounds - action, a_min = 0.0, a_max = np.inf))

        reward += penalty

        # Maximize the mass of product
        reward += float(20e3 - state[2])

        # Scale the reward
        reward *= 1 / 20e3

        # Adapt according to time step
        reward *= self.integration_settings.dt

        # Add the current time step to make the batch as short as possible
        reward += self.integration_settings.dt # NOTE: Maybe add later

        # Inverse signs to make it a reward and not a stage cost
        reward *= -1

        # Scale the reward
        reward *= 10
        return reward

class Anti_Windup_Controller:

    def __init__(self, state_sp: float, action_sp: float, K_P: float, K_I: float, action_lower: float, action_upper: float, dt: float, max_i_error: float = np.inf,):
        
        self.state_sp = state_sp
        self.action_sp = action_sp
        self.K_P = K_P
        self.K_I = K_I
        self.action_lower = action_lower
        self.action_upper = action_upper
        self.dt = dt

        self.integrated_error: float = 0.0
        self.integrated_anti_windup_error: float = 0.0
        self.max_i_error: float = max_i_error
        self.anti_windup: bool = False
        return
    
    def step(self, measurement: float):
        self.error = error = self.state_sp - measurement
        self.integrated_error += error * self.dt

        action = self.action_sp + self.K_P * error + self.K_I * self.integrated_error

        if self.anti_windup:
            self.integrated_error = np.clip(self.integrated_error, a_min = - np.abs(self.integrated_anti_windup_error), a_max = + np.abs(self.integrated_anti_windup_error))
            self.integrated_error = np.clip(self.integrated_error, a_min = - np.abs(self.max_i_error), a_max = + np.abs(self.max_i_error))

        if (action <  self.action_lower or action > self.action_upper) or (np.abs(self.integrated_error) > np.abs(self.max_i_error)):
            action = np.clip(action, a_min = self.action_lower, a_max = self.action_upper)

            self.integrated_anti_windup_error = self.integrated_error
            self.anti_windup = True
        else:
            self.anti_windup = False

        return float(action)
    
    def update_state_setpoint(self, state_sp: float):
        self.state_sp = state_sp
        return
    
    def update_action_setpoint(self, action_sp: float):
        self.action_sp = action_sp
        return
    
    def update_K_P(self, K_P: float):
        self.K_P = K_P
        return
    
    def update_K_I(self, K_I: float):
        # self.integrated_error *= K_I / self.K_I
        # self.integrated_anti_windup_error *= K_I / self.K_I
        self.K_I = K_I
        return
    
    def inherit_integrated_error(self, controller, measurement: float = None):
        plus_minus = 1
        # if measurement is not None:
        #     if self.state_sp > controller.state_sp and measurement > self.state_sp:
        #         plus_minus = -1
        # else:
        #     if self.state_sp > controller.state_sp:
        #         plus_minus = -1
        error_1 = self.state_sp - measurement
        error_2 = controller.state_sp - measurement
        # error_2 = controller.error
        inhertiated_integrated_error = controller.integrated_error
        inhertiated_integrated_error += error_2 * controller.dt

        reference_control_action = controller.action_sp + controller.K_P * error_2 + controller.K_I * inhertiated_integrated_error

        self.integrated_error = 1 / self.K_I * (reference_control_action - self.action_sp - self.K_P * error_1)
        self.integrated_anti_windup_error = 1 / self.K_I * (plus_minus * controller.integrated_anti_windup_error * controller.K_I + controller.action_sp - self.action_sp + controller.K_P * error_2 - self.K_P * error_1) 
        self.anti_windup = controller.anti_windup
        return
    
    def reset(self):
        self.integrated_error = 0.0
        self.integrated_anti_windup_error = 0.0
        self.anti_windup = False
        return

if __name__ == "__main__":
    # define evironment instance
    env = Poly_reactor_SB3()


    # First batch phase
    # Theta 1
    observation, reward, termination, truncation, info = env.step(50e3)

    # Theta 2
    observation, reward, termination, truncation, info = env.step(30e3)

    # Theta 3
    observation, reward, termination, truncation, info = env.step(273.15 + 89)

    # Theta 4
    observation, reward, termination, truncation, info = env.step(273.15 + 80)

    # Theta 5
    observation, reward, termination, truncation, info = env.step(0.5)


    # Second batch phase
    # Theta 6
    observation, reward, termination, truncation, info = env.step(-100)

    # Theta 7
    observation, reward, termination, truncation, info = env.step(20e3)

    # Theta 8
    observation, reward, termination, truncation, info = env.step(273.15 + 90)
    
    # Theta 9
    observation, reward, termination, truncation, info = env.step(273.15 + 70)

    # Theta 10
    observation, reward, termination, truncation, info = env.step(1.0)


    # Third batch phase
    # Theta 11
    observation, reward, termination, truncation, info = env.step(273.15 + 90)

    from plot_reactor_trajectories import plot_trajectory

    x_data = env.x_data
    u_data = env.u_data
    plot_trajectory(x_data, u_data)

    