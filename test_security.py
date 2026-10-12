import base64
import io
import os
import unittest
import tempfile
from unittest.mock import patch
from flask import Flask, jsonify
from PIL import Image
from security import configure_security, current_session_id, admin_required, save_validated_upload, is_admin_authenticated

class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FLASK_SECRET_KEY": "x" * 48, "SESSION_COOKIE_SECURE": "false", "ADMIN_USERNAME": "", "ADMIN_PASSWORD": ""})
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        configure_security(self.app)
        @self.app.get("/session")
        def sid():
            return jsonify(sid=current_session_id())
        @self.app.get("/admin")
        @admin_required
        def admin():
            return "ok"
        @self.app.post("/upload")
        def upload():
            from flask import request
            return jsonify(path=save_validated_upload(request.files["image"], self.tmp.name))
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()
        self.env.stop()

    def test_untrusted_host_rejected(self):
        self.assertEqual(self.client.get("/session", base_url="http://untrusted.example").status_code, 400)

    def test_request_rate_limit(self):
        for _ in range(60):
            self.client.post("/missing", headers={"Origin": "http://localhost"})
        self.assertEqual(self.client.post("/missing", headers={"Origin": "http://localhost"}).status_code, 429)

    def test_secret_required(self):
        with patch.dict(os.environ, {"FLASK_SECRET_KEY": ""}):
            with self.assertRaises(RuntimeError): configure_security(Flask("bad"))

    def test_session_stable_and_isolated(self):
        a = self.client.get("/session").json["sid"]
        self.assertEqual(a, self.client.get("/session").json["sid"])
        self.assertNotEqual(a, self.app.test_client().get("/session").json["sid"])

    def test_cookie_and_cache_headers(self):
        r = self.client.get("/session")
        self.assertIn("HttpOnly", r.headers["Set-Cookie"])
        self.assertIn("SameSite=Lax", r.headers["Set-Cookie"])
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    def test_admin_disabled_without_config(self):
        self.assertEqual(self.client.get("/admin").status_code, 403)

    def test_admin_authentication(self):
        with patch.dict(os.environ, {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "unique-test-password"}):
            self.assertEqual(self.client.get("/admin").status_code, 401)
            bad = base64.b64encode(b"admin:wrong").decode()
            self.assertEqual(self.client.get("/admin", headers={"Authorization": "Basic " + bad}).status_code, 401)
            good = base64.b64encode(b"admin:unique-test-password").decode()
            self.assertEqual(self.client.get("/admin", headers={"Authorization": "Basic " + good}).status_code, 200)

    def test_cross_origin_and_missing_origin_rejected(self):
        self.assertEqual(self.client.post("/upload").status_code, 403)
        self.assertEqual(self.client.post("/upload", headers={"Origin": "https://evil.example"}).status_code, 403)

    def test_legacy_static_paths_blocked(self):
        for route in ["/static/uploads/a.jpg", "/static/audio/a.wav", "/static/admin_uploads/a.jpg"]:
            self.assertEqual(self.client.get(route).status_code, 404)

    def test_invalid_upload_rejected(self):
        r = self.client.post("/upload", headers={"Origin": "http://localhost"}, data={"image": (io.BytesIO(b"not an image"), "image.jpg")})
        self.assertEqual(r.status_code, 400)

    def test_image_reencoded_with_safe_filename(self):
        stream = io.BytesIO()
        Image.new("RGB", (12, 12), "white").save(stream, "PNG")
        stream.seek(0)
        r = self.client.post("/upload", headers={"Origin": "http://localhost"}, data={"image": (stream, "../../escape.png")})
        self.assertEqual(r.status_code, 200)
        from pathlib import Path
        dest = Path(r.json["path"])
        self.assertEqual(dest.parent, Path(self.tmp.name))
        self.assertEqual(dest.suffix, ".jpg")
        with Image.open(dest) as img: self.assertEqual(img.format, "JPEG")

    def test_pixel_limit(self):
        stream = io.BytesIO()
        Image.new("RGB", (12, 12)).save(stream, "PNG")
        stream.seek(0)
        with patch("security.MAX_PIXELS", 100):
            r = self.client.post("/upload", headers={"Origin": "http://localhost"}, data={"image": (stream, "large.png")})
            self.assertEqual(r.status_code, 400)

    def test_body_size_limit(self):
        r = self.client.post("/upload", headers={"Origin": "http://localhost"}, data=b"x" * (8 * 1024 * 1024 + 1), content_type="application/x-www-form-urlencoded")
        self.assertEqual(r.status_code, 413)


class PatchedRouteTests(unittest.TestCase):
    """Exercise the actual patched routes without initializing ML/API clients."""
    def setUp(self):
        import ast
        from pathlib import Path
        from flask import request, session, abort, send_from_directory
        self.env = patch.dict(os.environ, {"FLASK_SECRET_KEY": "x" * 48, "SESSION_COOKIE_SECURE": "false", "ADMIN_USERNAME": "", "ADMIN_PASSWORD": ""})
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Flask("patched-routes")
        self.app.config["TESTING"] = True
        configure_security(self.app)
        names = {"chat", "clear_history", "mute", "serve_audio", "admin_clear", "admin_dashboard", "admin_evaluate"}
        tree = ast.parse(Path(__file__).with_name("app.py").read_text(encoding="utf-8"))
        selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        from werkzeug.exceptions import HTTPException
        import re
        self.g = dict(app=self.app, request=request, session=session, abort=abort, jsonify=jsonify,
            current_session_id=current_session_id, admin_required=admin_required, is_admin_authenticated=is_admin_authenticated, skin_class_names=["acne", "eczema"],
            send_from_directory=send_from_directory, AUDIO_DIR=self.tmp.name, audio_owners={},
            store={}, detected_disease_history={}, last_mentioned_disease={}, prediction_history=[],
            render_template=lambda _: "admin", clear_session_data=lambda sid: None,
            HTTPException=HTTPException, re=re, clean_response_for_display=lambda s: s,
            generate_speech_safe=lambda s: None, url_for=lambda *a, **k: "", tts_status=True)
        exec(compile(selected, "patched-app-routes", "exec"), self.g)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()
        self.env.stop()

    def test_greeting_sets_distinct_session_ids(self):
        for client in [self.client, self.app.test_client()]:
            r = client.post("/get", data={"msg": "hi"}, headers={"Origin": "http://localhost"})
            self.assertEqual(r.status_code, 200)
            self.assertIn("Hello", r.json["answer"])
        with self.client.session_transaction() as s: self.assertIn("sid", s)

    def test_clear_history_cannot_target_another_session(self):
        self.g["store"]["victim"] = "private"
        r = self.client.post("/clear_history", json={"session_id": "victim"}, headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.g["store"]["victim"], "private")

    def test_admin_routes_disabled(self):
        self.assertEqual(self.client.get("/admin").status_code, 403)
        self.assertEqual(self.client.post("/admin/clear", headers={"Origin": "http://localhost"}).status_code, 403)

    def test_admin_evaluation_is_protected(self):
        r = self.client.post("/admin/evaluate", data={"msg": "hi"}, headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 403)

    def test_public_labels_cannot_poison_metrics(self):
        r = self.client.post("/get", data={"msg": "hi", "actual_class": "not_a_model_class"}, headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 200)
        with patch.dict(os.environ, {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "test-password"}):
            auth = base64.b64encode(b"admin:test-password").decode()
            r = self.client.post("/admin/evaluate", data={"msg": "hi", "actual_class": "not_a_model_class"}, headers={"Origin": "http://localhost", "Authorization": "Basic " + auth})
            self.assertEqual(r.status_code, 400)

    def test_audio_ownership_enforced(self):
        from pathlib import Path
        Path(self.tmp.name, "sample.wav").write_bytes(b"demo")
        self.client.post("/get", data={"msg": "hi"}, headers={"Origin": "http://localhost"})
        with self.client.session_transaction() as s: sid = s["sid"]
        self.g["audio_owners"]["sample.wav"] = sid
        r = self.client.get("/audio/sample.wav")
        self.assertEqual(r.status_code, 200)
        r.close()
        self.assertEqual(self.app.test_client().get("/audio/sample.wav").status_code, 404)

if __name__ == "__main__":
    unittest.main(verbosity=2)
