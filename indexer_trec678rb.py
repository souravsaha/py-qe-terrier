import pyterrier as pt
from definitions.trec678rb import index_loc, collection_documents_dir

files = pt.io.find_files(collection_documents_dir)
indexer = pt.TRECCollectionIndexer(index_loc, verbose=True, blocks=False)
indexer.index(files)
