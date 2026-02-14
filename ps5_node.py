import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

# Make sure you are running this node in parallel!
# ros2 run joy joy_node --ros-args -p dev:=/dev/input/js2

def deadzone(x, dz=0.05):
    return 0.0 if abs(x) < dz else x

def offset(raw):
    return 0.5 * (1.0 - raw)

class PS5ToTau(Node):
    def __init__(self):
        super().__init__('ps5_to_tau')

        self.sub = self.create_subscription(Joy, '/joy', self.cb, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/tau_cmd', 10)

        # Scale correctly
        self.tau_max = [1.0, 1.0, 1.0]  # [surge, sway, yaw]

    def cb(self, msg: Joy):
        surge =  deadzone(msg.axes[1])
        sway  =  -deadzone(msg.axes[0])

        l2 = offset(msg.axes[2])   # L2 trigger
        r2 = offset(msg.axes[5])   # R2 trigger
        yaw = deadzone(r2 - l2)

        out = Float32MultiArray()
        out.data = [
            self.tau_max[0] * surge,
            self.tau_max[1] * sway,
            self.tau_max[2] * yaw
        ]
        self.pub.publish(out)

def main():
    rclpy.init()
    node = PS5ToTau()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
