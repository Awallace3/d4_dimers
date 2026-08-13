reinitialize
# A rigidly rotated copy of the source geometry gives a clear pi-stacking view.
load 24_Benzene-Benzene_pi-pi_1.00_presentation.xyz, benzene_dimer

# Ball-and-stick representation with conventional element colors.
hide everything
show sticks, benzene_dimer
show spheres, benzene_dimer
# Smaller atom spheres and slimmer bonds give a conventional molecular model.
set stick_radius, 0.10
set stick_h_scale, 0.70
set sphere_scale, 0.20
set valence, 1
color grey70, benzene_dimer and elem C
color white, benzene_dimer and elem H

# Presentation settings
bg_color white
# Preserve alpha transparency in the PNG background.
set ray_opaque_background, off
set antialias, 2
# Crisp, high-specular ray-traced material (less diffuse/cartoon-like).
set ambient, 0.22
set direct, 0.82
set specular, 0.72
set shininess, 75
set reflect, 0.18
set orthoscopic, on
set depth_cue, 0
set ray_shadows, 1
set ray_trace_mode, 1
set label_color, black

# Side-on, oblique view prevents the stacked benzenes from overlapping.
turn x, 72
turn y, -22
turn z, 10
zoom benzene_dimer, 2.0

ray 2400, 1800
png 24_Benzene-Benzene_pi-pi_1.00_pymol.png, dpi=300
save 24_Benzene-Benzene_pi-pi_1.00_pymol.pse
quit
