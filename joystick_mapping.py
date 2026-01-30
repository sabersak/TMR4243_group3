from dataclasses import dataclass

@dataclass(slots=True, init=False)
class JoystickMapping:
    LEFT_STICK_HORIZONTAL: int
    LEFT_STICK_VERTICAL: int
    RIGHT_STICK_HORIZONTAL: int
    RIGHT_STICK_VERTICAL: int
    LEFT_TRIGGER: int
    RIGHT_TRIGGER: int
    A_BUTTON: int
    B_BUTTON: int
    X_BUTTON: int
    Y_BUTTON: int