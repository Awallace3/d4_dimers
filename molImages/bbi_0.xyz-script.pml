load bbi_0.xyz

color grey, elem c
hide spheres,*
show stick,*
set_bond stick_radius, 0.2, v.
set stick_h_scale,1
zoom center,5
center v.
rotate x, 65
set ray_opaque_background, off
set opaque_background, off

ray 2000,2000
png bbi_0.xyz.png dpi=200
