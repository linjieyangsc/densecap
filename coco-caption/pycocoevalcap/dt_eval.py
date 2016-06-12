__author__ = 'ljyang'
from tokenizer.ptbtokenizer import PTBTokenizer
#
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
#import nltk
class DtEvalCap:
    def __init__(self, ref_caps, model_caps):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.ref = ref_caps
        self.pred = model_caps
        self.params = {'image_id': []}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
	  anno = {}
	  anno['image_id'] = imgId
	  anno['caption'] = self.ref[imgId]
          gts[imgId] = [anno]
	  anno_pred = {}
	  anno_pred['image_id'] = imgId
	  anno_pred['caption'] = self.pred[imgId]
          res[imgId] = [anno_pred]

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
           # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
           # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
