import numpy as np
import sgd

class EmbeddingModel:

    def __init__(self, vocabulary, embed_size = 300):
        self.embed_size = embed_size
        self.vocabulary = vocabulary
        self.mtx_embed =np.random.randn(len(self.vocabulary), self.embed_size).astype(np.float32) * 1e-2

    def update_embed(self,learning_rate, grad, vec):
        return (learning_rate * (grad/len(vec))/np.sqrt(self.mtx_grad[vec]))

    def update_grad(self, grad, vec):
        return np.square(grad/len(vec)) 

    def forward(self, anchor, truth, wrong):
        title_vec = sgd.doc_to_vec(anchor, self.mtx_embed)
        text_vec = sgd.doc_to_vec(truth, self.mtx_embed)
        incor_text_vec = sgd.doc_to_vec(wrong, self.mtx_embed)
        return title_vec,text_vec, incor_text_vec, sgd.calculate_loss(title_vec, text_vec, incor_text_vec)

    def backward(self, title_vec, text_vec, incor_text_vec, anchor, truth, wrong, learning_rate):
        grad_anchor, grad_truth, grad_wrong = sgd.calculate_gradient(title_vec, text_vec, incor_text_vec)
        #add vector to every row of matrix
        self.mtx_grad[anchor] += self.update_grad(grad_anchor, anchor)
        self.mtx_grad[truth] += self.update_grad(grad_truth, truth)
        self.mtx_grad[wrong] += self.update_grad(grad_wrong, wrong)

        #update matrix of embeddings
        self.mtx_embed[anchor] -= self.update_embed(learning_rate, grad_anchor, anchor)
        self.mtx_embed[truth] -= self.update_embed(learning_rate, grad_truth, truth)
        self.mtx_embed[wrong] -= self.update_embed(learning_rate, grad_wrong, wrong)

    def calculate_metric(self, val_titles, val_texts):
        #calculate the metric in the end of epoch
        number_of_docs = len(val_titles)
        mtx_title = sgd.get_document_term_sparse_mtx(val_titles, self.vocabulary)
        mtx_text = sgd.get_document_term_sparse_mtx(val_texts, self.vocabulary)
        mtx_title = (mtx_title/mtx_title.sum(axis=1))[:, np.newaxis]
        mtx_text = (mtx_text/mtx_text.sum(axis=1))[:, np.newaxis]
        mtx_title = mtx_title.dot(self.mtx_embed)
        mtx_text = mtx_text.dot(self.mtx_embed)
        res = mtx_title.dot(mtx_text.T)
        indexes_mtx = np.argsort(res, axis=1)
        correct_res = 0
        for idx, row in enumerate(indexes_mtx[:, -10:]):
            if (idx in row):
                correct_res += 1
        metric = correct_res/number_of_docs        
        #metric = np.mean(indexes == np.arange(number_of_docs))
        self.metric_res.append(metric)
        return metric


    
    def train(self, train_titles, train_texts, val_titles, val_texts, number_of_epoch = 2, learning_rate = 0.01, method = 'adagrad'):
        self.mtx_grad = np.full_like(self.mtx_embed, 1e-8, dtype=np.float32)            
        self.metric_res = []
        self.losses = []
        for i in range(number_of_epoch):
            train_wrong_texts = sgd.shuffle_text(train_texts)
            epoch_losses = []
            for anchor, truth, wrong in zip(train_titles, train_texts, train_wrong_texts):
                title_vec, text_vec, incor_text_vec, loss = self.forward(anchor, truth, wrong)
                epoch_losses.append(loss)
                if loss > 0:
                    self.backward(title_vec, text_vec, incor_text_vec, anchor, truth, wrong, learning_rate)
        #compare loss and metric with previous one        
            self.losses.append(np.mean(epoch_losses))        
        #current_metric = self.calculate_metric(val_titles, val_texts)
