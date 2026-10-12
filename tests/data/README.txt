Fixture for tests/assembly_regression.rs

ecoli_window.fasta
  NC_000913.3 (E. coli K-12 MG1655, GCF_000005845.2) positions 1,599,001-1,621,000: the window
  1,600,001-1,620,000, which shares no 31-mer with the rest of the genome, plus 1 kb of flank on
  each side (reads at the window's edges reach into it).

ecoli_window_1.fastq.gz, ecoli_window_2.fastq.gz
  The 2,072 read pairs of the Mason simulation simreads_mason_150_350_30_30_13478 (150 bp reads,
  ~30x) in which either mate has a 31-mer, sampled every 10 bases, that occurs in the window.
  Reads, qualities and order are unchanged from the simulation.
