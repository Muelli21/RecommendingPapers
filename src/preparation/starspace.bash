./starspace train \
    -trainFile "text.txt" \
    -model starspace_embedding \
    -trainMode 5 \
    -adagrad true \
    -ngrams 1 \
    -epoch 20 \
    -dim 200 \
    -similarity "cosine" \
    -minCount 75 \
    -verbose true \
    -negSearchLimit 15 \
    -lr 0.05 \
    -verbose true \
    -normalizeText true \