def refine_face(obj_file_path, new_file_path, to_del_face_idx, to_del_vertices_idx=[]):
    with open(obj_file_path, 'r') as file:
        with open(new_file_path, 'w') as new_file:
            for id, line in enumerate(file):
                if line.startswith('f ') and id in to_del_face_idx:
                    continue
                # if line.startswith('v ') and id in to_del_vertices_idx:
                #     continue
                else:
                    new_file.write(line)
                    
# def refine_face_ver2(obj_file_path, new_file_path, to_del_face_idx):
#     with open(obj_file_path, 'r') as file:
#         with open(new_file_path, 'w') as new_file:
#             for id, line in enumerate(file):
#                 if line.startswith('f ') and id in to_del_face_idx:

#                 else:
#                     new_file.write(line)


if __name__ == "__main__":
    # "_test/shapes/face_mp468_4.obj" # this one should be unmodified version
    # r_obj_file_path = "_test/shapes/face_mp468_4.obj"
    # w_new_file_path = "_test/shapes/face_mp468_5.obj" # this one should be final version
    
    r_obj_file_path = "_test/shapes/face_mp468_3.obj"
    w_new_file_path = "_test/shapes/face_mp468_7.obj"

    # deal with face not needed
    # ================== face not needed ==================
    to_del_face_idx_step1 = [6, 410, 412, 498, 500, 694, 696, 866]
    to_del_face_idx_step1_5 = [7, 411, 413, 499, 501, 695, 697, 867]
    # to_del_face_idx_step2 = [6, 409, 410, 495, 496, 689, 690, 859]
    
    to_del_face_idx_final_ver = [6, 7, 410, 411, 412, 413, 498, 499, 500, 501, 694, 695, 696, 697, 866, 867]
    
    to_del_face_idx = to_del_face_idx_final_ver

    # add 468 for all elements in to_del_face_idx
    to_del_face_idx = [x + 468 for x in to_del_face_idx]

    print(to_del_face_idx)
    # ================== face not needed ==================
    
    # deal with vertices not needed
    # ================== vertices not needed ==================
    # to_del_vertices_idx_ = [94, 128, 133, 235, 324, 357, 362, 455]
    # to_del_vertices_idx= [x-1 for x in to_del_vertices_idx_]

    # ================== vertices not needed ==================
    

    refine_face(r_obj_file_path, w_new_file_path, to_del_face_idx)



        
            
            
            
            
        
        