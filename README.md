# shape2motion-pytorch
* pts, `(N,9)` array of points of the object point cloud with xyz, rgb, normals.
* semantic_masks, `(N,1)` array, values are the part semantic IDs.
* instance_masks, `(N,1)` array of point class in range [0, K] of `N` points and `K` movable parts.
* joint_types, `(K,1)` array of the joint types
    * rotation joint: 0
    * translation joint: 1
* joint_origins, `(K,3)` array of the joint origins.
* joint_axes, `(K,3)` array of the joint axes.
* transformation_back, `(16,1)` transformation matrix transform the object back to origin pose in the scene.


```
yaml
# instance_id is "${scanID}_${objectId}"
instance_id:
attr:
  objectCat: object category
  objectId: object ID
  semanticId: object semantic ID
  numParts: number of movable parts
dataset:
  pts: object point cloud with xyz, rgba, normals. Shape [N, 10]
  semantic_masks: Part points semantic segmentation mask. Shape [N, 1].
  instance_masks: Part points instance segmentation mask. Shape [N, 1].
  joint_types: K joint types with value (0, 1). Shape [K, 1] 0: rotation, 1: translation
  joint_origins: K joint origins. Shape [K, 3]
  joint_axes: K joint axes. Shape [K, 3]
  transformation_back: transformation matrix transform the object back to origin pose in the scene. Shape [16, 1]
... other object instances
```
