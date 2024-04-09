import numpy as np 
from scipy.optimize import minimize
import json 


class Camera:
    def __init__(self, K, target_point, up, origin, time):
        self.origin = origin 
        self.up = up
        self.target_point = target_point
        self.K = K
        self.time = time
    
    def get_intrinsics(self):
        return self.K
        
    def get_extrinsics(self):
        forward = self.target_point - self.origin
        forward /= np.linalg.norm(forward)        
        right = np.cross(forward, self.up)
        right/= np.linalg.norm(right)
        up = np.cross(right, forward)
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = self.origin
        return view_matrix
    
    def __add__(self, other_camera):
        new_origin = self.origin + other_camera.origin
        new_up = self.up + other_camera.up
        new_target_point = self.target_point  + other_camera.target_point
        new_K = self.K + other_camera.K
        new_time = self.time + other_camera.time
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)

    def __mul__(self, scalar):
        new_origin = self.origin*scalar
        new_up = self.up*scalar
        new_target_point = self.target_point*scalar
        new_K = self.K*scalar
        new_time = self.time*scalar
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)
    
    def __rmul__(self, scalar):
        new_origin = self.origin*scalar
        new_up = self.up*scalar
        new_target_point = self.target_point*scalar
        new_K = self.K*scalar
        new_time = self.time*scalar
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)

    def __str__(self):
        ret = f"origin: {np.array2string(self.origin)}, \n tp: {np.array2string(self.target_point)}"
        return ret


class Trajectory:
    def __init__(self, cameras, interpolations, center=np.array([0, 0, 0])):
        self.cameras = cameras
        self.interpolations = interpolations
        self.trajectory = []
        self.trajectory_computed = False
        self.center = center

    
    def compute_trajectory(self):
        self.trajectory_computed = True
        
        for ind, camera in enumerate(self.cameras):
            # add current camera
            self.trajectory.append(camera)
            
            #check if last camera 
            if ind < len(self.cameras)-1:
                
                #get next camera
                next_camera = self.cameras[ind+1]
                (interpolation_type, interpolation_number) = self.interpolations[ind]
                
                if interpolation_type == "sphere":
                    segment = self.sphere_interpolation(camera, next_camera, interpolation_number, self.center)
                    self.trajectory += segment 
                
                if interpolation_type == "linear":
                    segment = self.linear_interpolation(camera, next_camera, interpolation_number)
                    self.trajectory += segment 
                    

    def linear_interpolation(self, c1, c2, num_cams):
        segment = []
        for i in range(num_cams):
            c = c1*((num_cams-i)/num_cams) + (i/num_cams)*c2
            segment.append(c)
        return segment
            
    def sphere_interpolation(self, c1, c2, num_cams, center):
        segment = []
        o1, o2 = c1.origin, c2.origin
        d1, d2 = np.linalg.norm(c1.origin-center), np.linalg.norm(c2.origin-center)
        starting_vect = o1 - center
        ending_vect = o2 - center

        angle_between = np.arccos(starting_vect.T @ ending_vect / (np.linalg.norm(starting_vect) * np.linalg.norm(ending_vect))) 
        angle_increments = angle_between/num_cams
        cross_vect = np.cross(ending_vect, starting_vect)
        cross_vect = cross_vect/np.linalg.norm(cross_vect)

        for i in range(num_cams):
            c = ((num_cams-i)/num_cams)*c1 + (i/num_cams)*c2
            camera_position = center + self.rodrigues_rotation(o1 - center, cross_vect, -angle_increments*i)
            distance = ((num_cams-i)/num_cams)*d1 + (i/num_cams)*d2
            forward = center - camera_position
            forward /= np.linalg.norm(forward)
            # print(camera_position)
            camera_position = center - distance * forward 
            c.origin = camera_position
            segment.append(c)
        return segment


    def add_camera(self, camera):
        self.cameras.append(camera)
        self.trajectory_computed = False

    
    def delete_camera(self, index):
        self.cameras.pop(index)
    
    def get_trajectory(self):
        if self.trajectory_computed:
            return self.trajectory
        else:
            self.compute_trajectory()
            return self.trajectory 
    
    def save_transforms(self, path, frames = None):
        if frames == None:
            frames = self.trajectory
        in_cam = self.trajectory[0]
        frames = [fr.get_extrinsics() for fr in frames]
        new_json = {}
        new_json["camera"] = in_cam.get_intrinsics()
        views = []
        for ind, f in enumerate(frames):
            frame = {}
            frame["filepath"] = f"{ind:04d}.h5"
            frame["transform_matrix"] = f.tolist()
            views.append(frame)

        new_json["frames"] = views
        
        json_object = json.dumps(new_json, indent=4)
        with open(path, 'w') as f:
            f.write(json_object)
        
    
    def rodrigues_rotation(self, n, u, theta):
        u_dot_n = u @ n
        u_cross_n = np.cross(u, n)
        return n * np.cos(theta) + u_cross_n * np.sin(theta) + u * u_dot_n * (1 - np.cos(theta))
        
    
    def smoothen_trajectory(self):
        points = np.array([f.origin for f in self.trajectory])
        times = np.array([f.time for f in self.trajectory])
        times = ((times-times.min())/(times.max()-times.min()))        
        
        
        def bezier_curve(params, t):
            t = t[:, None]
            P0, P1, P2, P3, P4 = params[:3][None], params[3:6][None], params[6:9][None], params[9:12][None], params[12:15][None]
            
            # P0 twice to the curve starts and ends at the same point
            return ((1 - t)**3) * P0 + 3 * t * ((1 - t)**2) * P1 + 3 * (t**2) * (1 - t) * P2 + (t**3) * P0 +  (t**2) * (1 - t)**3 * P3 +  (t)**3 * ((1 - t)**2 )* P4
            

        # Define the cost function
        def cost_function(params, points):
            theta = times
            helix_points = bezier_curve(params, theta)
            return np.sum((points - helix_points)**2)

        # Initial guess for parameters
        initial_guess = np.ones((15))

        # Minimize the cost function
        result = minimize(cost_function, initial_guess, args=(points,))
        best_params = result.x

        # Generate points on the bezier curve using best parameters
        theta = times
        helix_points = bezier_curve(best_params, theta)
        for ind, cam in enumerate(self.trajectory):
            cam.origin = helix_points[ind]
            self.trajectory[ind] = cam


def kennedy_smooth_traj():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.3, -3.73,  -0.814]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1500,
        K = K)
    
    c2 = Camera(
        origin=np.array([1, -3.01, 2.118]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1660,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.3, -0.65, 4 ]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1820,
        K = K,        
    )
    
    c4 = Camera(
        origin=np.array([-1, -2.869,  2.162]),
        up=np.array([-0.4, 0, 0.11]), 
        # target_point=np.array([0.4, 1.3, 0.7]),
        target_point=np.array([1.1, 0.1, -0.4]),
        time=1980,
        K = K,        
    )
    
    c5 = Camera(
        origin=np.array([0.3, -3.73,  -0.814]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=2140,
        K = K,        
    )
    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("linear", 160), ("sphere", 160), ("sphere", 160), ("sphere", 160)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    traj.save_transforms("")
    return t


def mirror_smooth_traj():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([1.23, -0.3, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1650,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.4, -2.62, 2.446]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1900,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.4, 1.59, 3.2 ]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2000,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([1.23, -0.3, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2200,
        K = K)
    
    
    cameras = [c1, c2, c3, c4]
    interpolations = [("sphere", 250), ("sphere", 100), ("sphere", 200)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t


def grating_smooth_trajectory():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([1.230, -0.673, 3.562]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1600,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.6, -2.62, 2.446]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1750,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.5, 1.695, 3.273 ]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, 0.2, -0.4]),
        time=2050,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([1.23, -0.67, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2200,
        K = K)
    

    
    cameras = [c1, c2, c3, c4]
    interpolations = [("sphere", 150), ("sphere", 300), ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t



def coke_smooth_trajectory_old():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1650,
        K = K)
    
    c2 = Camera(
        origin=np.array([0, 2.781, 2.0]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1800,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([-0.6, -0.84, -3.3]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1850,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([0.2, -4.2, 1]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1950,
        K = K)
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2100,
        K = K)

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 150), ("sphere", 100), ("sphere", 100),  ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    # traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t


def coke_smooth_trajectory():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1550,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.1, -3.8, 1]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1700,
        K = K)
    
    c3 = Camera(
        origin=np.array([-0.4, -0.84, -3.3]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1850,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0, 2.781, 2.0]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2000,
        K = K,        
    )
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2150,
        K = K)
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 150), ("sphere", 100), ("sphere", 100),  ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    return t


def coke_one_wide():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1500,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1600,
        K = K)
    
    c3 = Camera(
        origin=np.array([1.07, 1.696,  3.27]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1880,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2200,
        K = K,        
    )
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2300,
        K = K)

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 100), ("sphere", 280), ("sphere", 320),  ("sphere", 100)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    
    traj.save_transforms("", t)    
    return t


def coke_unwarped():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1500-280,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1600-280,
        K = K)
    
    c3 = Camera(
        origin=np.array([1.07, 1.696,  3.27]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1880-280,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2200-280,
        K = K,        
    )
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2300-280,
        K = K)


    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 100), ("sphere", 280), ("sphere", 320),  ("sphere", 100)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    return t


if __name__=="__main__":
    pass