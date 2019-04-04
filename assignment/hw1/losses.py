import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        num_train = x.shape[0]
        j = torch.arange(num_train)
        scores_y_i = x_scores.gather(1, y.unsqueeze(1)) # s_{y_i}
        mat = scores_y_i.repeat(1, x_scores.shape[1]) # matrix where row i is s_{y_i}
        M = x_scores - mat + self.delta
        zeros = torch.zeros(M.shape)
        margins = torch.max(zeros, M) # max{0, s_j - s_{y_i} + delta}
        fix = torch.zeros(num_train)
        margins[j, y] = fix # fix values for j = y_i (set to 0)
        loss = torch.sum(margins) / num_train # calculate average
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['margins'] = margins
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['num_train'] = num_train
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        margins = self.grad_ctx['margins']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        num_train = self.grad_ctx['num_train']
        G = margins
        G[margins > 0] = 1 # G holds 1 where margins is greater than 0
        row_sum = torch.sum(G, 1)
        G[torch.arange(num_train), y] = -row_sum # The sum should ignore the right classification
        grad = torch.mm(torch.t(x), G) # X^T * G
        grad /= num_train # Should normalizie
        #dW += reg*W
        # ========================

        return grad
