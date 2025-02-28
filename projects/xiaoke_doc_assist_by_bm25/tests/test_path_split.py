import  os

file_path = r"C:\Users\k\Documents\project\programming_project\python_project\development\XiaokeAILabs\projects\xiaoke_doc_assist_by_bm25\tests\test_datas\test_mineru_add_image_description_如何阅读一本书.pdf"
file_name = os.path.basename(file_path)
print(file_name)  
splitext = os.path.splitext(file_name)

print(splitext) 