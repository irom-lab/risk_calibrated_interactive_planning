import pybullet as p
import time

# Connect to the PyBullet physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)  # Set gravity

# Load ground plane
p.loadURDF("plane.urdf")

# Create first car
car1 = p.loadURDF("racecar/racecar.urdf", [-2, 0, 0.1], useFixedBase=False)

# Create second car
car2 = p.loadURDF("racecar/racecar.urdf", [2, 0, 0.1], useFixedBase=False)

# Create spring constraint between the two cars
spring_constraint = p.createConstraint(car1, -1, car2, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0],
                                       [0, 0, 0], childFrameOrientation=[0, 0, 0, 1])

# Parameters for the spring
spring_length = 4.0
spring_stiffness = 1000.0
spring_damping = 10.0

# Simulation loop
for _ in range(1000):
    p.stepSimulation()

    # Get positions of the two cars
    pos1, _ = p.getBasePositionAndOrientation(car1)
    pos2, _ = p.getBasePositionAndOrientation(car2)

    # Compute the spring force
    delta_pos = [pos2[i] - pos1[i] for i in range(3)]
    spring_force = [spring_stiffness * delta_pos[i] - spring_damping * p.getJointState(car1, spring_constraint)[6 + i]
                    for i in range(3)]

    # Apply forces to the cars
    p.applyExternalForce(car1, -1, spring_force, [0, 0, 0], p.LINK_FRAME)
    p.applyExternalForce(car2, -1, [-x for x in spring_force], [0, 0, 0], p.LINK_FRAME)

    time.sleep(1. / 240.)  # Control simulation speed

# Disconnect from the physics server
p.disconnect()
