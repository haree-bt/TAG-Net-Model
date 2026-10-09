import mujoco

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.5">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 400, 400)
    renderer.update_scene(data)
    img = renderer.render()
    print("SUCCESS! Rendered shape:", img.shape)
except Exception as e:
    print("ERROR:", e)
