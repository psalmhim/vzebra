import pygame
import random

# Initialize pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Define screen size
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sea World")

# Define the physical dimensions of the environment (in meters)
physical_width = 20  # meters
physical_height = 15  # meters

# Calculate the scale factor (pixels per meter)
scale_factor = WIDTH / physical_width

class Camera:
    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + WIDTH // 2
        y = -target.rect.centery + HEIGHT // 2

        # Limit scrolling to map size
        x = min(0, x)  # Left
        y = min(0, y)  # Top
        x = max(-(self.width - WIDTH), x)  # Right
        y = max(-(self.height - HEIGHT), y)  # Bottom

        self.camera = pygame.Rect(x, y, self.width, self.height)

class Fish(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30, 10))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.pos = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))

    def update(self):
        self.pos += self.velocity
        self.rect.topleft = (self.pos.x, self.pos.y)
        # Wrap around screen for demo purposes
        if self.rect.left > physical_width * scale_factor:
            self.rect.right = 0
        if self.rect.right < 0:
            self.rect.left = physical_width * scale_factor
        if self.rect.top > physical_height * scale_factor:
            self.rect.bottom = 0
        if self.rect.bottom < 0:
            self.rect.top = physical_height * scale_factor

# Create a group for all sprites and a fish
all_sprites = pygame.sprite.Group()
fish = Fish(100, 100)
all_sprites.add(fish)

# Create the camera
camera = Camera(physical_width * scale_factor, physical_height * scale_factor)

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update
    all_sprites.update()
    camera.update(fish)

    # Draw
    screen.fill(WHITE)
    for sprite in all_sprites:
        screen.blit(sprite.image, camera.apply(sprite))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
