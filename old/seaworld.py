import pygame
import random
from vzt_basic import *
from vzt_zebra import *
from vzt_seaworld import *
import time
import copy

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1600, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sea World")

# Create subscreens
main_view_rect = pygame.Rect(0, 0, WIDTH * 2 // 3, HEIGHT)
detail_view_rect = pygame.Rect(WIDTH * 2 // 3, 0, WIDTH // 2, HEIGHT)

main_view = pygame.Surface(main_view_rect.size)
detail_view = pygame.Surface(detail_view_rect.size)

# Create the SeaWorld instance
sea_world = SeaWorld(main_view, WIDTH * 2 // 3, HEIGHT)
sea_world.load_sea_plants()
sea_world.load_planktons()

# Initialize fish
zfishes = [
    ZFish(
        (random.randint(-100, 100), random.randint(-100, 100)),
        (random.uniform(-1, 1), random.uniform(-1, 1)),
        #[random.uniform(1, 1), random.uniform(0, 0)],
        #[random.uniform(-1, -1), random.uniform(0, 0)], #left 
        #[random.uniform(0, 0), random.uniform(1, 1)], #left 
    ) 
    for _ in range(1)
]

# Select one fish for detailed view
detailed_fish = zfishes[0]

# Set up a dictionary to track the next update time for each fish
next_update_time = {zfish: time.time() + random.uniform(0.5, 2.0) for zfish in zfishes}

# Main loop
running = True
clock = pygame.time.Clock()
ct = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the sea world
    sea_world.update()

    # Clear the main view and detail view
    # main_view.fill(Color.BLUE)
    detail_view.fill(Color.LIGHTBLUE)

    sea_world.draw()  # Ensure this is being called and works correctly

    # Update and draw fish in the main view
    current_time = time.time()
    for zfish in zfishes:
        if current_time >= next_update_time[zfish]:
            zfish.update_random_firing()
            next_update_time[zfish] = current_time + random.uniform(
                0.05, 0.5
            )  # Set the next update time
            zfish.update_position()
        zfish.draw(main_view)
        #print(f"Fish position: {zfish.morph.pos}, Fish rect: {zfish.rect}")

    # Draw detailed view of the selected fish
    # zfish_detail = copy.deepcopy(detailed_fish)
    # zfish_detail.morph.pos = [-300, 50]
    # zfish_detail.update_body()
    # zfish_detail.draw(detail_view)

    # Draw the subscreens onto the main screen
    screen.blit(main_view, main_view_rect.topleft)
    # screen.blit(detail_view, detail_view_rect.topleft)

    # Flip the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
