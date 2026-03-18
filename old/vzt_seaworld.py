import pygame
import os
import random

# Base Object class
class IObject(pygame.sprite.Sprite):
    def __init__(self, x=0, y=0, image=None, morph=None, viz=None, state=None):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = image
        self.image = image
        self.rect = self.image.get_rect(center=(x, y)) if image else None
        self.size = self.rect.size if self.rect else (0, 0)
        self.angle = 0

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)


# SeaPlant class
class SeaPlant(IObject):
    def __init__(
        self,
        x,
        y,
        image,
        draw_boundary=False,
        rotation_mode="random",
        global_angle=0,
        global_direction=1,
    ):
        super().__init__(x, y, image)
        self.rect = self.image.get_rect(
            midbottom=(x, y)
        )  # Set the bottom center of the rect to (x, y)
        self.angle_direction = 1
        self.draw_boundary = draw_boundary
        self.rotation_mode = rotation_mode
        self.global_angle = global_angle
        self.global_direction = global_direction

    def update(self):
        if self.rotation_mode == "random":
            # Rotate left and right randomly
            self.angle += self.angle_direction * 0.5
            if abs(self.angle) >= 15:  # Change direction at 15 degrees
                self.angle_direction *= -1
        elif self.rotation_mode == "synchronized":
            # Use global rotation settings
            self.angle = self.global_angle

        # Rotate around the midbottom
        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 1
        )  # Apply rotation
        self.rect = self.image.get_rect(
            midbottom=self.rect.midbottom
        )  # Keep the midbottom fixed

    def draw(self, screen):
        super().draw(screen)
        if 0:
            # Draw the rotation center
            pygame.draw.circle(screen, (255, 0, 0), self.rect.midbottom, 5)
            # Draw the image boundary if flag is set
            if self.draw_boundary:
                # Get the points of the rotated rectangle
                rect_points = [
                    self.rect.topleft,
                    self.rect.topright,
                    self.rect.bottomright,
                    self.rect.bottomleft,
                ]
                # Calculate the rotated points
                rotated_points = []
                for point in rect_points:
                    offset = pygame.Vector2(point) - self.rect.center
                    rotated_offset = offset.rotate(
                        -self.angle
                    )  # Rotate the points in the same direction as the image
                    rotated_point = self.rect.center + rotated_offset
                    rotated_points.append(rotated_point)
                # Draw the rotated rectangle
                pygame.draw.polygon(screen, (0, 255, 0), rotated_points, 1)


# Plankton class
class Plankton(IObject):
    def __init__(self, x, y, image):
        super().__init__(x, y, image)
        self.initial_position = pygame.Vector2(x, y)
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(
            random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
        )

    def update(self):
        # Random walk: add a small random vector to the current velocity
        self.velocity += pygame.Vector2(
            random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
        )
        self.position += self.velocity
        distance_from_initial = (self.position - self.initial_position).length()

        # Keep plankton within a range of 50 pixels from the initial position
        if distance_from_initial > 50:
            # Reverse direction and add noise between -45 to 45 degrees
            current_angle = self.velocity.angle_to(pygame.Vector2(1, 0))
            new_angle = current_angle + 180 + random.uniform(-45, 45)
            self.velocity = self.velocity.rotate(new_angle - current_angle)
            self.position += self.velocity  # Correct the position

        # Random rotation
        self.angle += random.uniform(-5, 5)
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)


# SeaWorld class
class SeaWorld:
    def __init__(self, screen, width, height, scale_factor=0.2):
        self.screen = screen
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.plants_rotation_mode = "random"
        self.plants_global_angle = 0
        self.plants_global_direction = 1  # left/right

        self.background_image = pygame.image.load(
            os.path.join("seaworld", "background.png")
        ).convert()
        self.background_image = pygame.transform.scale(
            self.background_image, (width, height)
        )
        self.all_sprites = pygame.sprite.Group()
        self.sea_plants = pygame.sprite.Group()
        self.planktons = pygame.sprite.Group()

    def load_sea_plants(self):
        sea_plant_images = []
        for i in range(1, 20):
            image = pygame.image.load(
                os.path.join("seaworld", f"seaplant-{i}.png")
            ).convert_alpha()
            image = pygame.transform.scale(
                image,
                (
                    int(image.get_width() * self.scale_factor),
                    int(image.get_height() * self.scale_factor),
                ),
            )
            sea_plant_images.append(image)

        for _ in range(19):
            image = random.choice(sea_plant_images)
            image_width, image_height = image.get_width(), image.get_height()
            x, y = random.randint(0, self.width - image_width), random.randint(
                image_height, self.height
            )
            sea_plant = SeaPlant(x, y, image, draw_boundary=True)
            self.all_sprites.add(sea_plant)
            self.sea_plants.add(sea_plant)

    def load_planktons(self):
        plankton_images = []
        for i in range(1, 16):
            image = pygame.image.load(
                os.path.join("seaworld", f"plankton-{i}.png")
            ).convert_alpha()
            image = pygame.transform.scale(
                image,
                (
                    int(image.get_width() * self.scale_factor),
                    int(image.get_height() * self.scale_factor),
                ),
            )
            plankton_images.append(image)
        for _ in range(15):
            image = random.choice(plankton_images)
            image_width, image_height = image.get_width(), image.get_height()
            x, y = random.randint(0, self.width - image_width), random.randint(
                0, self.height - image_height
            )
            plankton = Plankton(x, y, image)
            self.all_sprites.add(plankton)
            self.planktons.add(plankton)

    def update(self):
        # Update global rotation settings if in synchronized mode
        if self.plants_rotation_mode == "synchronized":
            self.plants_global_angle += self.plants_global_direction * 0.5
            if abs(self.plants_global_angle) >= 15:  # Change direction at 15 degrees
                self.global_direction *= -1
        for sea_plant in self.sea_plants:
            if self.plants_rotation_mode == "synchronized":
                sea_plant.global_angle = self.plants_global_angle
                sea_plant.global_direction = self.plants_global_direction
            sea_plant.update()

        self.planktons.update()

    def draw(self):
        self.screen.blit(
            self.background_image, [0, 0]
        )  # Draw the stretched background image
        self.all_sprites.draw(self.screen)  # Draw all sprites on top of the background
        for sea_plant in self.sea_plants:
            sea_plant.draw(self.screen)
