"""
Prey capture kinematics: J-turn → approach swim → capture strike.

Zebrafish larvae capture prey (paramecia) with a stereotyped motor
sequence (Bianco & Engert 2015):
  1. J-turn: orient toward prey (slow, precise turn, 3 steps)
  2. Approach swim: close distance with fine alignment (5 steps max)
  3. Capture strike: fast lunge (2 steps, committed trajectory)

Triggered when: FORAGE goal + food visible (≥5 px) + close (< 80 px).
Aborted if: food lost from view or obstacle blocks path.
"""


class PreyCaptureKinematics:
    def __init__(self):
        self.phase = 'NONE'  # NONE, J_TURN, APPROACH, STRIKE
        self.timer = 0
        self.total_strikes = 0
        self.strike_active = False

    def update(self, goal, food_px, food_distance, food_lateral_bias,
               obstacle_px=0) -> tuple:
        """
        Returns: (turn, speed) override or None if not in capture sequence.
        goal: 0=FORAGE
        food_px: total food pixels visible
        food_distance: estimated distance to nearest food (from binocular or pixel size)
        food_lateral_bias: -1 (left) to +1 (right)
        obstacle_px: obstacle pixels (abort if too many)
        """
        if self.phase == 'NONE':
            # Trigger: foraging + food visible + close enough
            if goal == 0 and food_px >= 5 and food_distance < 80:
                self.phase = 'J_TURN'
                self.timer = 3
            return None

        # Abort conditions
        if food_px < 2:
            self.phase = 'NONE'
            self.strike_active = False
            return None
        if obstacle_px > 30:
            self.phase = 'NONE'
            self.strike_active = False
            return None

        if self.phase == 'J_TURN':
            # Orient toward prey: slow precise turn
            self.timer -= 1
            if self.timer <= 0:
                self.phase = 'APPROACH'
                self.timer = 5
            return (food_lateral_bias * 0.8, 0.3)  # slow turn toward food

        if self.phase == 'APPROACH':
            # Close distance with fine alignment
            self.timer -= 1
            if food_distance < 35:
                self.phase = 'STRIKE'
                self.timer = 2
                self.total_strikes += 1
                self.strike_active = True
            elif self.timer <= 0:
                self.phase = 'NONE'
                return None
            return (food_lateral_bias * 0.5, 0.5)  # moderate approach

        if self.phase == 'STRIKE':
            # Fast lunge — committed trajectory
            self.timer -= 1
            if self.timer <= 0:
                self.phase = 'NONE'
                self.strike_active = False
            return (food_lateral_bias * 0.3, 1.5)  # fast lunge

        return None

    def reset(self):
        self.phase = 'NONE'
        self.timer = 0
        self.strike_active = False
