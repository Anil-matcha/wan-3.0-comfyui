import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is None:

    @unittest.skip("Contract tests require the PyTorch runtime bundled with ComfyUI.")
    class Wan30NodeContractTests(unittest.TestCase):
        def test_comfyui_runtime_is_available(self):
            pass

else:
    from wan_3_nodes import (
        Wan30ImageEdit,
        Wan30ImageToVideo,
        Wan30TextToImage,
        Wan30TextToVideo,
        _output_value,
    )

    _IMAGE = torch.zeros(1, 2, 2, 3)
    _OUTPUT_IMAGE = torch.zeros(1, 2, 2, 3)
    _OUTPUT_FRAME = torch.zeros(1, 2, 2, 3)

    class Wan30NodeContractTests(unittest.TestCase):
        def test_processing_prediction_url_is_not_treated_as_media(self):
            self.assertIsNone(
                _output_value(
                    {"status": "processing", "output": {"urls": {"get": "https://api.test/prediction"}}}
                )
            )
            self.assertEqual(
                _output_value({"outputs": ["https://cdn.test/image.png"]}),
                "https://cdn.test/image.png",
            )

        @patch("wan_3_nodes._download_image", return_value=_OUTPUT_IMAGE)
        @patch("wan_3_nodes._poll", return_value={"output": {"image": "https://image.test/t2i.png"}})
        @patch("wan_3_nodes._submit", return_value="t2i-request")
        def test_text_to_image_uses_current_endpoint_and_payload(self, submit, _poll, _download):
            result = Wan30TextToImage().run("A cinematic sunrise", api_key="test-key")

            self.assertIs(result[0], _OUTPUT_IMAGE)
            self.assertEqual(result[1], "https://image.test/t2i.png")
            self.assertEqual(submit.call_args.args[0], "test-key")
            self.assertEqual(submit.call_args.args[1], "wan3.0-text-to-image")
            self.assertEqual(submit.call_args.args[2], {"prompt": "A cinematic sunrise"})

        @patch("wan_3_nodes._upload_image", return_value="https://image.test/source.jpg")
        @patch("wan_3_nodes._download_image", return_value=_OUTPUT_IMAGE)
        @patch("wan_3_nodes._poll", return_value={"outputs": ["https://image.test/edited.png"]})
        @patch("wan_3_nodes._submit", return_value="edit-request")
        def test_image_edit_uploads_source_image(self, submit, _poll, _download, upload):
            Wan30ImageEdit().run(
                "Replace the background with a warm studio",
                api_key="test-key",
                image=_IMAGE,
            )

            upload.assert_called_once_with("test-key", _IMAGE)
            self.assertEqual(submit.call_args.args[1], "wan3.0-image-edit")
            self.assertEqual(
                submit.call_args.args[2],
                {
                    "prompt": "Replace the background with a warm studio",
                    "image_url": "https://image.test/source.jpg",
                },
            )

        @patch("wan_3_nodes._first_frame", return_value=_OUTPUT_FRAME)
        @patch("wan_3_nodes._poll", return_value={"video": "https://video.test/t2v.mp4"})
        @patch("wan_3_nodes._submit", return_value="t2v-request")
        def test_text_to_video_uses_current_endpoint_and_payload(self, submit, _poll, _frame):
            result = Wan30TextToVideo().run("A cinematic sunrise", api_key="test-key")

            self.assertEqual(result[0], "https://video.test/t2v.mp4")
            self.assertEqual(submit.call_args.args[1], "wan3.0-text-to-video")
            self.assertEqual(submit.call_args.args[2], {"prompt": "A cinematic sunrise"})

        @patch("wan_3_nodes._upload_image", return_value="https://image.test/start.png")
        @patch("wan_3_nodes._first_frame", return_value=_OUTPUT_FRAME)
        @patch("wan_3_nodes._poll", return_value={"outputs": ["https://video.test/i2v.mp4"]})
        @patch("wan_3_nodes._submit", return_value="i2v-request")
        def test_image_to_video_sends_image_url(self, submit, _poll, _frame, upload):
            Wan30ImageToVideo().run(
                "The camera slowly pushes in",
                api_key="test-key",
                image=_IMAGE,
            )

            upload.assert_called_once_with("test-key", _IMAGE)
            self.assertEqual(submit.call_args.args[1], "wan3.0-image-to-video")
            self.assertEqual(
                submit.call_args.args[2],
                {
                    "prompt": "The camera slowly pushes in",
                    "image_url": "https://image.test/start.png",
                },
            )


if __name__ == "__main__":
    unittest.main()
