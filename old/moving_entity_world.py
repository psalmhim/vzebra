import torch, random, math

class MovingEntityWorld:
    """
    Produces prey or predator with different size, speed,
    motion patterns, and looming trajectory.
    prey: small, darting, local
    predator: large, looming, fast approach
    """
    def __init__(self, w=64, h=64, prey_prob=0.5):
        self.w = w
        self.h = h
        self.prey_prob = prey_prob
        self.reset()

    def reset(self):
        # choose prey or predator
        self.type = "prey" if random.random() < self.prey_prob else "predator"
        self.size = 3 if self.type=="prey" else 8
        self.speed = 0.8 if self.type=="prey" else 1.6

        # random spawn point
        self.x = random.uniform(8, self.w-8)
        self.y = random.uniform(8, self.h-8)
        theta = random.uniform(0,2*math.pi)
        self.vx = self.speed * math.cos(theta)
        self.vy = self.speed * math.sin(theta)

    def step(self):
        self.x += self.vx
        self.y += self.vy

        # bounce and re-randomize predator to simulate approach
        if self.x<5 or self.x>self.w-5:
            self.vx *= -1
        if self.y<5 or self.y>self.h-5:
            self.vy *= -1

        frame = torch.zeros(1, self.w*self.h)
        x0 = int(self.x)
        y0 = int(self.y)
        s = self.size//2

        # draw square
        for i in range(-s, s+1):
            for j in range(-s, s+1):
                xi = min(max(x0+i,0), self.w-1)
                yi = min(max(y0+j,0), self.h-1)
                frame[0, yi*self.w + xi] = 1.0

        label = 0 if self.type=="prey" else 1  # 0=prey, 1=predator
        return frame, label
