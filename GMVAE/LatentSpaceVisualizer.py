import pygame as pg
from models.linear_vae import LinearVariationalAutoencoder
import torch

vae = LinearVariationalAutoencoder()
vae.load_state_dict(torch.load("models/cc_noc_30e_3lf.pth"))

s1 = vae.sample_from_prior(batch_size=1)

z1 = s1["z"].reshape(-1)
print(z1.shape)
print(z1)

# WIP, want to easily be able to manipulate latent values, and see a generation

pg.init()

w, h = 500, 300
font = pg.font.Font(pg.font.get_default_font(), 14)
scr = pg.display.set_mode((w, h))
bg = pg.Surface((w, h))

latent_variables = 10
latent_values = [0.0 for _ in range(latent_variables)]
latent_range = 2.5

bg.fill((255, 255, 255))
for i in range(latent_variables):
    variableText = font.render(str(i), 0, (0, 0, 0))
    bg.blit(variableText, (320, 10 + 22 * (i + 1)))
    pg.draw.line(bg, (0, 0, 0), (340, 18 + 22 * (i + 1)), (w - 20, 18 + 22 * (i + 1)))

run = True
while run:
    scr.blit(bg, (0, 0))
    pg.draw.rect(scr, (0, 0, 0), (10, 10, 28 * 10, 28 * 10))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                run = False
    pg.display.update()

