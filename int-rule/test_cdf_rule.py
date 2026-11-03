import unittest

import torch

from cdf_rule import calibrated_probs, predict_cdf_rule, train_cdf_rule
from utils_compare import model_reconstruction_kl


class CDFRuleTests(unittest.TestCase):
    def test_reconstruction_kl_value_and_target_gradient(self):
        logits = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64, requires_grad=True)
        p = logits.softmax(dim=1)
        q = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
        kl = model_reconstruction_kl(p, q)
        expected = (p * (p.log() - q.log())).sum(dim=1).mean()
        self.assertTrue(torch.allclose(kl, expected))
        self.assertTrue(torch.allclose(model_reconstruction_kl(q, q), torch.tensor(0., dtype=q.dtype)))
        grad = torch.autograd.grad(kl, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.allclose(grad, expected_grad))
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_threshold_identity_and_interpolation(self):
        p = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
        eta = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(calibrated_probs(p, eta, eta, torch.tensor(1.0)), p))
        midpoint = torch.logit((torch.sigmoid(eta[0]) + torch.sigmoid(eta[1])) / 2)
        out = calibrated_probs(p, eta, midpoint.reshape(1), torch.tensor(1.0))
        self.assertTrue(torch.allclose(out, torch.tensor([[0.2, 0.8]], dtype=p.dtype)))

    def test_four_variants_and_gradients(self):
        torch.manual_seed(7)
        H = torch.randn(12, 3, 4)
        p = torch.softmax(H.mean(1), dim=1)
        target = torch.softmax(torch.randn(12, 4), dim=1)
        for internal in (False, True):
            for plus in (False, True):
                result = train_cdf_rule(p, target, H=H if internal else None,
                    plus=plus, num_outer_iters=2, num_steps_A=3, num_steps_B=3,
                    num_steps_C=3)
                if internal:
                    p_eval = torch.softmax((H * result['w'][None, :, None]).sum(1), 1)
                else:
                    p_eval = p
                pred, labels, scores = predict_cdf_rule(p_eval, result)
                self.assertTrue(torch.isfinite(pred).all())
                self.assertTrue(torch.allclose(pred.sum(1), torch.ones(12), atol=1e-6))
                self.assertEqual(labels.shape, scores.shape)
                self.assertTrue(torch.all(torch.diff(result['eta']) > 0))

    def test_gradients_reach_probability_and_thresholds(self):
        p = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
        eta = torch.tensor([-1., 0., 1.], requires_grad=True)
        alpha = torch.tensor([-.5, .5, 1.5], requires_grad=True)
        beta = torch.tensor(1., requires_grad=True)
        out = calibrated_probs(p, eta, alpha, beta)
        loss = -(torch.tensor([[.4, .3, .2, .1]]) * out.log()).sum()
        loss.backward()
        for value in (p, eta, alpha, beta):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
