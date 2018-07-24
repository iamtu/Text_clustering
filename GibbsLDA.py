# https://shuyo.wordpress.com/2011/05/24/collapsed-gibbs-sampling-estimation-for-latent-dirichlet-allocation-1/
import numpy as np
class GibbsLDA(object):
    def __init__(self, docs, word2int, n_topics=2, alpha=0.01, beta=0.01, iter=2000):
        self.alpha = alpha
        self.beta = beta
        self.K = n_topics
        
        self.word2int = word2int #dictionary word:idx
        self.int2word = {}
        for word in word2int:
            self.int2word[word2int[word]] = word
                
        self.V = len(word2int)
        self.docs = docs #corpus
        
        self.z_m_n = [] # topic of word n of document m: we need to sample.
        self.n_m_z = np.zeros((len(self.docs), self.K)) + self.alpha     # number of time topic z occurs in document m
        self.n_z_t = np.zeros((self.K, self.V)) + self.beta # number of time word t assigned to topic z
        self.n_z = np.zeros(self.K) + self.V * self.beta    # number of time words assigned topic z in all corpus
        self.MAX_ITER = iter
        
        print 'LDA model: #docs = ', len(self.docs), ' - n_topics = ', self.K, ' - alpha = ', self.alpha, \
                ' - beta - ', self.beta, ' - V = ', self.V

    def inference(self):
        print 'Gibbs sampling inference'
        # ----- init phase
        for m, doc in enumerate(self.docs):
            z_n = []
            for t in doc:
                # draw from the posterior
                p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))

        # --- inference 
        for iter in xrange(self.MAX_ITER):
                print 'iter : ', iter
            for m, doc in enumerate(self.docs):
                for n, t in enumerate(doc):
                    # discount for n-th word t with topic z
                    z = self.z_m_n[m][n]
                    self.n_m_z[m, z] -= 1
                    self.n_z_t[z, t] -= 1
                    self.n_z[z] -= 1

                    # sampling new topic
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z # Here is only changed.
                    new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                    # preserve the new topic and increase the counters
                    self.z_m_n[m][n] = new_z
                    self.n_m_z[m, new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1
        # --- estimate word-topic distribution
        self.phi = self.n_z_t / self.n_z[:,np.newaxis]
        # ---- estimate topic-document distribution
        self.theta = self.n_m_z / self.n_m_z.sum(axis=1).reshape((len(self.docs),1))
    
    def top_topics(n_top_words):
        top_words = []
        sorted_phi = np.sort(self.phi, axis = 1)
        top_idxes = sorted_phi[:,0:n_top_words]
        for topic in top_idxes:
            top_word = []
            for word_idx in topic:
                top_word.append(self.int2word[word_idx])
            top_words.append(top_word)
        return top_words




if __name__ == '__main__':
    X = np.array([
        [0, 0, 1, 2, 2],
        [0, 0, 1, 1, 1],
        [0, 1, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [3, 3, 4, 4, 4],
        [3, 4, 4, 4, 4]
    ])
    W = np.array([0, 1, 2, 3, 4])
    model = GibbsLDA(X, W)
    model.inference()
    print model.phi
    print model.theta
