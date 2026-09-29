import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "wr2" / "descriptions" / "wr2"


def actuator_order() -> list[str]:
    return [
        line.strip()
        for line in (ROBOT_DIR / "actuator_order.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.parse(ROBOT_DIR / "wr2.xml").getroot()

    def test_actuator_order(self):
        actual = [element.get("name") for element in self.root.findall("actuator/*")]
        self.assertEqual(actual, actuator_order())

    def test_servo_defaults_and_imu_sensors_are_present(self):
        self.assertIsNotNone(self.root.find(".//default[@class='htd45hServo']"))
        actual_sensors = {
            element.get("name") for element in self.root.findall("sensor/*")
        }
        self.assertEqual(
            actual_sensors,
            {
                "torso_imu_gyro",
                "torso_imu_accel",
                "torso_imu_mag",
                "torso_imu_quat",
            },
        )

    def test_waist_transmission(self):
        equality = self.root.find(
            "equality/joint[@joint1='waist_yaw_driven'][@joint2='waist_yaw_drive']"
        )
        self.assertIsNotNone(equality)
        self.assertEqual(equality.get("polycoef"), "0 1.0 0 0 0")

    def test_named_foot_contacts(self):
        for side in ("left", "right"):
            self.assertIsNotNone(self.root.find(f".//body[@name='{side}_foot']"))
            geom = self.root.find(f".//geom[@name='{side}_foot_collision']")
            self.assertIsNotNone(geom)
            self.assertEqual(geom.get("type"), "box")
            self.assertIsNotNone(self.root.find(f".//site[@name='{side}_foot_center']"))

    def test_zero_angle_home_pose(self):
        torso = self.root.find("worldbody/body[@name='torso']")
        self.assertIsNotNone(torso)
        self.assertEqual(torso.get("pos"), "0 0 0.305000")

        key = self.root.find("keyframe/key[@name='home']")
        self.assertIsNotNone(key)
        qpos = [float(value) for value in key.get("qpos", "").split()]
        ctrl = [float(value) for value in key.get("ctrl", "").split()]
        self.assertEqual(len(qpos), 25)
        self.assertEqual(len(ctrl), 17)
        self.assertEqual(qpos[:7], [0.0, 0.0, 0.305, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(value == 0.0 for value in qpos[7:]))
        self.assertTrue(all(value == 0.0 for value in ctrl))

    def test_walk_home_pose(self):
        key = self.root.find("keyframe/key[@name='walk_home']")
        self.assertIsNotNone(key)
        qpos = [float(value) for value in key.get("qpos", "").split()]
        ctrl = [float(value) for value in key.get("ctrl", "").split()]
        self.assertEqual(qpos[2], 0.304)
        self.assertEqual(len(qpos), 25)
        self.assertEqual(len(ctrl), 17)
        self.assertEqual(ctrl[actuator_order().index("left_knee_pitch")], -0.24)
        self.assertEqual(ctrl[actuator_order().index("right_knee_pitch")], 0.24)

    def test_mjx_variant_is_sensor_free_and_keeps_the_contract(self):
        root = ET.parse(ROBOT_DIR / "wr2_mjx.xml").getroot()
        self.assertIsNone(root.find("sensor"))
        actual = [element.get("name") for element in root.findall("actuator/*")]
        self.assertEqual(actual, actuator_order())
        self.assertIsNotNone(root.find("keyframe/key[@name='walk_home']"))


if __name__ == "__main__":
    unittest.main()
