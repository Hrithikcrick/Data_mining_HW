GSPAN_EXE=/home/sibgat/SJaiswal/A1/pkg/q2_final/gSpan6/gSpan
FSG_EXE=/home/sibgat/SJaiswal/A1/pkg/q2_final/fsg/Linux/fsg
GASTON_EXE=/home/sibgat/SJaiswal/A1/pkg/q2_final/gaston-1.1/gaston

DATASET=/home/sibgat/SJaiswal/A1/dataset/q2_final/Yeast/167.txt_graph
OUTDIR=/home/sibgat/SJaiswal/A1/q2_final/output

cd /home/sibgat/SJaiswal/A1/q2_final

mkdir -p logs "$OUTDIR"

bash new_q2_final.sh \
    "$GSPAN_EXE" \
    "$FSG_EXE" \
    "$GASTON_EXE" \
    "$DATASET" \
    "$OUTDIR"
