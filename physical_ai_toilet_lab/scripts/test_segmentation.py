import hashlib,json,unittest
import numpy as np
import torch
from segmentation_model import ROOT,TinyUNet,metric


class SegmentationTests(unittest.TestCase):
    def test_metrics_known_examples(self):
        truth=np.array([[[1,0],[0,0]],[[0,0],[0,0]]],dtype=np.uint8);roi=np.ones((2,2),bool)
        perfect=metric(truth==1,truth,roi)
        self.assertEqual(perfect['iou'],1.);self.assertEqual(perfect['empty_images_with_any_false_positive'],0)
        missed=metric(np.zeros_like(truth,bool),truth,roi);self.assertEqual(missed['recall'],0.)
        pred=truth.astype(bool);pred[1,0,0]=True
        false=metric(pred,truth,roi);self.assertEqual(false['iou'],.5);self.assertEqual(false['empty_images_with_any_false_positive'],1)

    def test_metrics_exclude_unobserved_pixels(self):
        truth=np.array([[[1,255],[0,0]]],dtype=np.uint8)
        pred=np.array([[[True,True],[False,True]]])
        roi=np.array([[True,True],[True,False]])
        scores=metric(pred,truth,roi)
        self.assertEqual(scores['iou'],1.);self.assertEqual(scores['fp'],0)
        empty=metric(np.zeros((1,2,2),bool),np.zeros((1,2,2),np.uint8),roi)
        self.assertIsNone(empty['positive_image_mean_iou'])

    def test_model_input_output_shape(self):
        torch.set_num_threads(2)
        with torch.no_grad():out=TinyUNet()(torch.zeros(2,3,240,320))
        self.assertEqual(tuple(out.shape),(2,240,320));self.assertTrue(torch.isfinite(out).all())

    def test_frozen_weights_match_evaluation(self):
        folder=ROOT/'output/learning_v1';frozen=json.loads((folder/'frozen_config.json').read_text())
        result=json.loads((folder/'evaluation.json').read_text())
        digest=hashlib.sha256((folder/'best_model.pt').read_bytes()).hexdigest()
        self.assertEqual(digest,frozen['model_sha256']);self.assertEqual(digest,result['model_sha256'])
        self.assertEqual(frozen['threshold'],result['threshold']);self.assertFalse(frozen['test_accessed'])

    def test_adapter_is_rgb_only_and_keeps_roi(self):
        from learned_perception import LearnedDirtDetector
        from PIL import Image
        from segmentation_model import DATA
        detector=LearnedDirtDetector('cpu')
        rgb=np.array(Image.open(DATA/'validation/200000_rgb.png'))
        mask=detector.predict(rgb);self.assertEqual(mask.shape,(240,320))
        self.assertFalse((mask&~detector.roi).any())
        with self.assertRaises(ValueError):detector.predict(np.zeros((20,20,3),dtype=np.uint8))
        with self.assertRaises(ValueError):detector.predict(rgb.astype(float))


if __name__=='__main__':unittest.main()
