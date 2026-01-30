import time
import numpy as np
import pygame
from TA_caseA import ThrustAllocator 

TAU_MAX = np.array([1.0, 1.0, 1.0], dtype=float)  # Magnitude of tau when keyboard is pressed

def main():
    pygame.init()
    pygame.display.set_mode((200, 200))  # Opens window

    ta = ThrustAllocator()

    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Arrow keys -> surge/sway
        surge = 1.0 if keys[pygame.K_UP] else (-1.0 if keys[pygame.K_DOWN] else 0.0)
        sway  = 1.0 if keys[pygame.K_RIGHT] else (-1.0 if keys[pygame.K_LEFT] else 0.0)

        # Q/E -> yaw
        yaw   = 1.0 if keys[pygame.K_e] else (-1.0 if keys[pygame.K_q] else 0.0)

        tau_cmd = TAU_MAX * np.array([surge, sway, yaw], dtype=float)

        F_cmd, alpha_cmd, u_cmd = ta.allocate(tau_cmd)

        print(f"tau={tau_cmd}  u={u_cmd}  F={F_cmd}  alpha={alpha_cmd}")
        time.sleep(0.05)

if __name__ == "__main__":
    main()
