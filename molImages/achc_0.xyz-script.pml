load achc_0.xyz

color grey, elem c
hide spheres,*
show stick,*
set_bond stick_radius, 0.2, v.
set stick_h_scale,1
zoom center,5
center v.
bg white

ray 2000,2000
png achc_0.xyz.png
