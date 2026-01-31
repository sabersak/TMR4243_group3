import numpy as np
import pygame

TAU_MAX = np.array([1.0, 1.0, 1.0], dtype=float)


class KeyboardLogic:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((200, 200))  # window for key events

    def get_tau_cmd(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        surge = 1.0 if keys[pygame.K_UP] else (-1.0 if keys[pygame.K_DOWN] else 0.0)
        sway  = 1.0 if keys[pygame.K_RIGHT] else (-1.0 if keys[pygame.K_LEFT] else 0.0)
        yaw   = 1.0 if keys[pygame.K_e] else (-1.0 if keys[pygame.K_q] else 0.0)

        return TAU_MAX * np.array([surge, sway, yaw], dtype=float)
