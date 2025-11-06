import unittest
import tempfile
import json
import os
from distributed_kv_manager.config_loader import load_config_from_json, dict_to_namespace

class ConfigLoaderTests(unittest.TestCase):
    def test_load_config_from_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.json")
            data = {"kv_transfer_config": {"storage_type": "local", "local_dir": "x"}, "rank": 3}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            ns = load_config_from_json(path)
            self.assertEqual(ns.rank, 3)
            self.assertEqual(ns.kv_transfer_config.storage_type, "local")

    def test_dict_to_namespace_nested(self):
        d = {"a": 1, "b": {"c": 2, "d": [ {"x": 10}, 5 ] }}
        ns = dict_to_namespace(d)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b.c, 2)
        self.assertEqual(ns.b.d[0].x, 10)
        self.assertEqual(ns.b.d[1], 5)

if __name__ == "__main__":
    unittest.main()
