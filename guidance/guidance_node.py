#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import rclpy
import rclpy.node
import rcl_interfaces.msg
import sensor_msgs.msg
import tmr4243_interfaces.msg

from stationkeeping import stationkeeping_reference
from straight_line import StraightLineGuidance, trigger_to_01, wrap_pi

def deadzone(x: float, dz: float = 0.08) -> float:
    return 0.0 if abs(x) < dz else x

class Guidance(rclpy.node.Node):
    TASK_STATIONKEEPING = 'stationkeeping'
    TASK_STRAIGHT_LINE  = 'straight_line'
    TASKS = [TASK_STATIONKEEPING, TASK_STRAIGHT_LINE]

    UPDATES = ['tracking', 'gradient', 'normalized', 'filtered']

    def __init__(self):
        super().__init__("tmr4243_guidance")

        self.pub_ref = self.create_publisher(
            tmr4243_interfaces.msg.Reference, '/tmr4243/control/reference', 1
        )

        self.sub_obs = self.create_subscription(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/eta', self.observer_callback, 10
        )

        # Joystick for runtime commands (Task 7.2)
        self.sub_joy = self.create_subscription(
            sensor_msgs.msg.Joy, '/joy', self.joy_callback, 10
        )
        self.last_joy = None
        self.prev_buttons = None

        self.last_observation = None

        # ---------------- Parameters ----------------
        self.task = Guidance.TASK_STATIONKEEPING
        self.declare_parameter(
            'task', self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="stationkeeping or straight_line",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                additional_constraints=f"Allowed values: {' '.join(Guidance.TASKS)}"
            )
        )

        # Stationkeeping reference (can be overridden by joystick "hold" action)
        self.declare_parameter('x_ref', 0.0)
        self.declare_parameter('y_ref', 0.0)
        self.declare_parameter('psi_ref', np.deg2rad(0.0)) 

        # Straight-line waypoints
        self.declare_parameter('p0', [0.0, 0.0])
        self.declare_parameter('p1', [3.0, 0.0])
        self.declare_parameter('clamp_segment', True)

        # Guidance update-law parameters
        self.declare_parameter(
            'update', 'normalized',
            rcl_interfaces.msg.ParameterDescriptor(
                description="tracking|gradient|normalized|filtered",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                additional_constraints=f"Allowed values: {' '.join(Guidance.UPDATES)}"
            )
        )
        self.declare_parameter('Uref', 0.0)   # m/s along-path (slow)
        self.declare_parameter('mu', 0.0)     # start with 0
        self.declare_parameter('eps', 1e-3)
        self.declare_parameter('lam', 2.0)

        # Activation sigma: if 0 => s_dot=0 (Task 7.2 requirement)
        self.declare_parameter('sigma', 0.0)

        # Joystick enable + mapping
        self.declare_parameter('joystick_enable', True)

        # Buttons (indices)
        self.declare_parameter('A_BUTTON', 0)  # DS4 Cross
        self.declare_parameter('B_BUTTON', 1)  # DS4 Circle
        self.declare_parameter('Y_BUTTON', 2)  # DS4 Triangle
        self.declare_parameter('X_BUTTON', 3)  # DS4 Square

        # Axes used for continuous tuning
        self.declare_parameter('RIGHT_STICK_HORIZONTAL', 3)
        self.declare_parameter('LEFT_TRIGGER', 2)
        self.declare_parameter('RIGHT_TRIGGER', 5)

        # Rates for joystick tuning
        self.declare_parameter('psi_rate', 0.1)   # rad/s per full stick deflection
        self.declare_parameter('U_rate', 0.05)    # (m/s)/s per full trigger deflection
        self.declare_parameter('mu_step', 0.2)    # toggle mu between 0 and this

        # Guidance internal state
        p0 = np.array(self.get_parameter('p0').value, dtype=float)
        p1 = np.array(self.get_parameter('p1').value, dtype=float)
        self.guidance = StraightLineGuidance(p0=p0, p1=p1)

        self.last_time = self.get_clock().now()

        self.timer = self.create_timer(0.1, self.guidance_loop)

        self.last_printed_psi_ref = float(self.get_parameter('psi_ref').value)
        self.last_printed_Uref = float(self.get_parameter('Uref').value)
        self.declare_parameter('stick_deadzone', 0.08)
        self.declare_parameter('trigger_deadzone', 0.05)

    def observer_callback(self, msg: tmr4243_interfaces.msg.Observer):
        self.last_observation = msg

    
    def print_guidance_status(self, reason: str):
        task = self.get_parameter('task').get_parameter_value().string_value
        x_ref = float(self.get_parameter('x_ref').value)
        y_ref = float(self.get_parameter('y_ref').value)
        psi_ref = float(self.get_parameter('psi_ref').value)

        p0 = np.array(self.get_parameter('p0').value, dtype=float)
        p1 = np.array(self.get_parameter('p1').value, dtype=float)

        update = self.get_parameter('update').get_parameter_value().string_value
        Uref = float(self.get_parameter('Uref').value)
        mu = float(self.get_parameter('mu').value)
        sigma = float(self.get_parameter('sigma').value)
        eps = float(self.get_parameter('eps').value)
        lam = float(self.get_parameter('lam').value)

        s_now = float(self.guidance.s) if hasattr(self.guidance, 's') else 0.0

        self.get_logger().info(
            f"[{reason}] "
            f"task={task}, "
            f"x_ref={x_ref:.3f}, y_ref={y_ref:.3f}, psi_ref={psi_ref:.3f}, "
            f"p0=[{p0[0]:.3f}, {p0[1]:.3f}], "
            f"p1=[{p1[0]:.3f}, {p1[1]:.3f}], "
            f"update={update}, Uref={Uref:.3f}, mu={mu:.3f}, sigma={sigma:.1f}, "
            f"eps={eps:.4f}, lam={lam:.3f}, s={s_now:.3f}"
        )

        if self.last_observation is not None:
            eta_hat = np.array(self.last_observation.eta, dtype=float).reshape(3)
            self.get_logger().info(
                f"[{reason}] eta_hat="
                f"[{eta_hat[0]:.3f}, {eta_hat[1]:.3f}, {eta_hat[2]:.3f}]"
            )

    def joy_callback(self, msg: sensor_msgs.msg.Joy):
        self.last_joy = msg

        if not bool(self.get_parameter('joystick_enable').value):
            return

        # Rising-edge detection for buttons
        buttons = list(msg.buttons)
        if self.prev_buttons is None:
            self.prev_buttons = buttons
            return

        A = int(self.get_parameter('A_BUTTON').value)
        B = int(self.get_parameter('B_BUTTON').value)
        X = int(self.get_parameter('X_BUTTON').value)
        Y = int(self.get_parameter('Y_BUTTON').value)

        def rising(i: int) -> bool:
            if i < 0 or i >= len(buttons) or i >= len(self.prev_buttons):
                return False
            return (buttons[i] == 1 and self.prev_buttons[i] == 0)

        # A: toggle task (stationkeeping <-> straight_line)
        if rising(A):
            current_task = self.get_parameter('task').get_parameter_value().string_value
            new_task = Guidance.TASK_STRAIGHT_LINE if current_task == Guidance.TASK_STATIONKEEPING else Guidance.TASK_STATIONKEEPING
            self.set_parameters([rclpy.parameter.Parameter('task', rclpy.Parameter.Type.STRING, new_task)])

            # When entering stationkeeping: "hold position" at current estimate
            if new_task == Guidance.TASK_STATIONKEEPING and self.last_observation is not None:
                eta_hat = np.array(self.last_observation.eta, dtype=float).reshape(3)
                self.set_parameters([
                    rclpy.parameter.Parameter('x_ref', rclpy.Parameter.Type.DOUBLE, float(eta_hat[0])),
                    rclpy.parameter.Parameter('y_ref', rclpy.Parameter.Type.DOUBLE, float(eta_hat[1])),
                    rclpy.parameter.Parameter('psi_ref', rclpy.Parameter.Type.DOUBLE, float(wrap_pi(eta_hat[2]))),
                    rclpy.parameter.Parameter('sigma', rclpy.Parameter.Type.DOUBLE, 0.0),
                ])

            # When entering maneuvering: enable sigma and reset s
            if new_task == Guidance.TASK_STRAIGHT_LINE:
                self.guidance.reset(0.0)
                self.set_parameters([rclpy.parameter.Parameter('sigma', rclpy.Parameter.Type.DOUBLE, 1.0)])

            self.task = new_task
            self.print_guidance_status("A pressed")

        # B: reset s and omega (and hold position if stationkeeping)
        if rising(B):
            self.guidance.reset(0.0)
            if self.task == Guidance.TASK_STATIONKEEPING and self.last_observation is not None:
                eta_hat = np.array(self.last_observation.eta, dtype=float).reshape(3)
                self.set_parameters([
                    rclpy.parameter.Parameter('x_ref', rclpy.Parameter.Type.DOUBLE, float(eta_hat[0])),
                    rclpy.parameter.Parameter('y_ref', rclpy.Parameter.Type.DOUBLE, float(eta_hat[1])),
                    rclpy.parameter.Parameter('psi_ref', rclpy.Parameter.Type.DOUBLE, float(wrap_pi(eta_hat[2]))),
                ])
            self.print_guidance_status("B pressed")

        # X: toggle sigma (maneuvering activation)
        if rising(X):
            sigma = float(self.get_parameter('sigma').value)
            sigma_new = 0.0 if sigma > 0.5 else 1.0
            self.set_parameters([rclpy.parameter.Parameter('sigma', rclpy.Parameter.Type.DOUBLE, sigma_new)])
            self.print_guidance_status("X pressed")

        # Y: toggle mu between 0 and mu_step
        if rising(Y):
            mu = float(self.get_parameter('mu').value)
            mu_step = float(self.get_parameter('mu_step').value)
            mu_new = 0.0 if mu > 1e-9 else mu_step
            self.set_parameters([rclpy.parameter.Parameter('mu', rclpy.Parameter.Type.DOUBLE, mu_new)])
            self.print_guidance_status("Y pressed")

        self.prev_buttons = buttons

    def guidance_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        dt = max(dt, 1e-3)

        # Read task param
        self.task = self.get_parameter('task').get_parameter_value().string_value

        # Allow continuous tuning from joystick (psi_ref and Uref)
        if bool(self.get_parameter('joystick_enable').value) and self.last_joy is not None:
            axes = self.last_joy.axes

            i_rs_h = int(self.get_parameter('RIGHT_STICK_HORIZONTAL').value)
            i_l2   = int(self.get_parameter('LEFT_TRIGGER').value)
            i_r2   = int(self.get_parameter('RIGHT_TRIGGER').value)

            psi_rate = float(self.get_parameter('psi_rate').value)
            U_rate   = float(self.get_parameter('U_rate').value)
            stick_deadzone = float(self.get_parameter('stick_deadzone').value)
            trigger_deadzone = float(self.get_parameter('trigger_deadzone').value)

            if 0 <= i_rs_h < len(axes):
                rs = deadzone(float(axes[i_rs_h]), stick_deadzone)
                if rs != 0.0:
                    dpsi = psi_rate * rs * dt
                    psi_ref = float(self.get_parameter('psi_ref').value)
                    psi_ref = wrap_pi(psi_ref + dpsi)
                    self.set_parameters([
                        rclpy.parameter.Parameter('psi_ref', rclpy.Parameter.Type.DOUBLE, psi_ref)
                    ])

            Uref = float(self.get_parameter('Uref').value)
            l2_raw = trigger_to_01(axes[i_l2]) if 0 <= i_l2 < len(axes) else 0.0
            r2_raw = trigger_to_01(axes[i_r2]) if 0 <= i_r2 < len(axes) else 0.0

            l2 = 0.0 if l2_raw < trigger_deadzone else l2_raw
            r2 = 0.0 if r2_raw < trigger_deadzone else r2_raw

            dU = U_rate * (r2 - l2) * dt
            if dU != 0.0:
                Uref = float(Uref + dU)
                self.set_parameters([rclpy.parameter.Parameter('Uref', rclpy.Parameter.Type.DOUBLE, Uref)])
        
        # if bool(self.get_parameter('joystick_enable').value) and self.last_joy is not None:
        #     axes = self.last_joy.axes

        #     i_rs_h = int(self.get_parameter('RIGHT_STICK_HORIZONTAL').value)
        #     i_l2   = int(self.get_parameter('LEFT_TRIGGER').value)
        #     i_r2   = int(self.get_parameter('RIGHT_TRIGGER').value)

        #     psi_rate = float(self.get_parameter('psi_rate').value)
        #     U_rate   = float(self.get_parameter('U_rate').value)

        #     if 0 <= i_rs_h < len(axes):
        #         dpsi = psi_rate * float(axes[i_rs_h]) * dt
        #         psi_ref = float(self.get_parameter('psi_ref').value)
        #         psi_ref = wrap_pi(psi_ref + dpsi)
        #         self.set_parameters([rclpy.parameter.Parameter('psi_ref', rclpy.Parameter.Type.DOUBLE, psi_ref)])

        #     # triggers: increase/decrease Uref
        #     Uref = float(self.get_parameter('Uref').value)
        #     l2 = trigger_to_01(axes[i_l2]) if 0 <= i_l2 < len(axes) else 0.0
        #     r2 = trigger_to_01(axes[i_r2]) if 0 <= i_r2 < len(axes) else 0.0
        #     dU = U_rate * (r2 - l2) * dt
        #     Uref = float(Uref + dU)
        #     self.set_parameters([rclpy.parameter.Parameter('Uref', rclpy.Parameter.Type.DOUBLE, Uref)])

        # Publish reference
        msg = tmr4243_interfaces.msg.Reference()

        if self.task == Guidance.TASK_STATIONKEEPING:
            x_ref = float(self.get_parameter('x_ref').value)
            y_ref = float(self.get_parameter('y_ref').value)
            psi_ref = float(self.get_parameter('psi_ref').value)

            eta_d, eta_ds, eta_ds2 = stationkeeping_reference(x_ref, y_ref, psi_ref)
            msg.eta_d = eta_d.flatten().tolist()
            msg.eta_ds = eta_ds.flatten().tolist()
            msg.eta_ds2 = eta_ds2.flatten().tolist()
            msg.w = 0.0
            msg.v_s = 0.0
            msg.v_ss = 0.0
            self.pub_ref.publish(msg)
            return

        # Straight-line maneuvering
        p0 = np.array(self.get_parameter('p0').value, dtype=float)
        p1 = np.array(self.get_parameter('p1').value, dtype=float)
        # Update path if user changed waypoints
        if np.linalg.norm(self.guidance.path.p0 - p0) > 1e-9 or np.linalg.norm(self.guidance.path.p1 - p1) > 1e-9:
            self.guidance = StraightLineGuidance(p0=p0, p1=p1)

        if self.last_observation is None:
            return

        eta_hat = np.array(self.last_observation.eta, dtype=float).reshape(3)
        p_hat = eta_hat[0:2]

        psi_ref = float(self.get_parameter('psi_ref').value)
        Uref = float(self.get_parameter('Uref').value)
        mu = float(self.get_parameter('mu').value)
        eps = float(self.get_parameter('eps').value)
        lam = float(self.get_parameter('lam').value)
        update = self.get_parameter('update').get_parameter_value().string_value
        clamp_segment = bool(self.get_parameter('clamp_segment').value)
        sigma = float(self.get_parameter('sigma').value)

        eta_d, eta_ds, eta_ds2, w, v_s, v_ss, _s = self.guidance.update(
            p_hat=p_hat, psi_ref=psi_ref,
            Uref=Uref, mu=mu, eps=eps, lam=lam,
            update_mode=update, dt=dt,
            clamp_segment=clamp_segment, sigma=sigma
        )
        psi_ref_now = float(self.get_parameter('psi_ref').value)
        Uref_now = float(self.get_parameter('Uref').value)

        if abs(psi_ref_now - self.last_printed_psi_ref) > 0.02:
            self.last_printed_psi_ref = psi_ref_now
            self.get_logger().info(f"[joystick] psi_ref={psi_ref_now:.3f} rad")

        if abs(Uref_now - self.last_printed_Uref) > 0.005:
            self.last_printed_Uref = Uref_now
            self.get_logger().info(f"[joystick] Uref={Uref_now:.3f} m/s")        

        msg.eta_d = eta_d.flatten().tolist()
        msg.eta_ds = eta_ds.flatten().tolist()
        msg.eta_ds2 = eta_ds2.flatten().tolist()
        msg.w = float(w)
        msg.v_s = float(v_s)
        msg.v_ss = float(v_ss)
        self.pub_ref.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Guidance()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()