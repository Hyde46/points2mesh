
class NeighborhoodDensitySubSample(RNGDataFlow):
    """docstring for NeighborhoodDensitySubSample

    Example:
        ds = dataset_train
        ds = NeighborhoodDensitySubSample(ds,neighborhood_sizes=32,sample_sizes=[1024,512,128])
        print ds.attributes()
        ds = PrintData(ds)
        ds.reset_state()

        ds.reset_state()

        for dp in ds.get_data():
            break

        datapoint 0<1 with 15 components consists of
          0: ndarray:float32 of shape (1024, 3) in range [-0.926731228828, 0.911605358124]
          1: ndarray:float32 of shape (1024, 1) in range [1.0, 1.0]
          2: ndarray:uint32 of shape (1024, 1) in range [33, 33]
          3: ndarray:int64 of shape (1024, 32) in range [0, 1023]
          4: ndarray:uint32 of shape (1024,) in range [0, 1595]
          5: ndarray:float32 of shape (512, 3) in range [-0.926731228828, 0.911605358124]
          6: ndarray:float32 of shape (512, 1) in range [1.0, 1.0]
          7: ndarray:uint32 of shape (512, 1) in range [33, 33]
          8: ndarray:int64 of shape (512, 32) in range [0, 511]
          9: ndarray:uint32 of shape (512,) in range [0, 812]
          10: ndarray:float32 of shape (128, 3) in range [-0.926731228828, 0.911605358124]
          11: ndarray:float32 of shape (128, 1) in range [1.0, 1.0]
          12: ndarray:uint32 of shape (128, 1) in range [33, 33]
          13: ndarray:int64 of shape (128, 32) in range [0, 127]
          14: ndarray:uint32 of shape (128,) in range [1, 333]

    """

    def __init__(
            self,
            ds,
            neighborhood_sizes,
            sample_sizes=None,
            strides=[1],
            num_points=None,
            sample_resolution=100000,
            global_density=False,
            time_dataflow=False):
        """Summary

        Args:
            ds: incoming dataflow
            neighborhood_sizes: either an integer of list of integers specifying the neighborhood sizes
            sample_sizes: list of sampled point cloud size of the outcome per level. If not used take strides instead.
            strides: either an integer of list of integers specifying the strides. Overridden by defining sample_sizes
            num_points: If strides is used num_points needs to be set
            global_density: define density only on first level

            TODO:
                - shuffle
                - scipy KDE
                - radii
        """

        super(NeighborhoodDensitySubSample, self).__init__()

        if sample_sizes is not None:
            if type(neighborhood_sizes) is list:
                assert len(sample_sizes) == len(neighborhood_sizes)
                self.neighborhood_sizes = neighborhood_sizes
            else:
                self.neighborhood_sizes = np.repeat(neighborhood_sizes,
                                                    len(sample_sizes))

            self.sample_sizes = sample_sizes
        else:
            assert type(strides) is list
            assert num_points is not None

            if type(neighborhood_sizes) is list:
                assert len(strides) == len(neighborhood_sizes)
                self.neighborhood_sizes = neighborhood_sizes
            else:
                self.neighborhood_sizes = np.repeat(neighborhood_sizes,
                                                    len(strides))
            self.sample_sizes = np.repeat(num_points, len(strides)) // strides

        # assert sample_resolution > self.sample_sizes[0], "sample_resolution needs to be higher than the number of points"

        self.ds = ds
        self.sample_resolution = sample_resolution
        self.global_density = global_density
        self.time_dataflow = time_dataflow

#         self.shuffle = shuffle

    def get_data(self):

        for dp in self.ds.get_data():
            location, feature, label = dp
            assert len(
                dp
            ) == 3, "incoming dp should only have location, feature, label, but len is %i" % len(
                dp)
            num_elements = location.shape[0]
            idxs = list(range(num_elements))

            current_location = dp[0]
            current_feature = dp[1]
            current_label = dp[2]

            cum_density = None

            ret = []

            N = len(current_location)
            assert self.sample_sizes[0] <= N

            if self.time_dataflow:
                start_time = timeit.default_timer()

            kdt = KDTree(current_location, leaf_size=16, metric='euclidean')

            if self.time_dataflow:
                elapsed = timeit.default_timer() - start_time
                print('kdt build: elapsed: ', elapsed)

                start_time = timeit.default_timer()

            # print('DensityNN query %i from %i' % (self.neighborhood_sizes[0], location.shape[0]))

            current_dist, current_neighborhood = kdt.query(
                current_location,
                k=self.neighborhood_sizes[0],
                dualtree=False,
                return_distance=True)

            if self.time_dataflow:
                elapsed = timeit.default_timer() - start_time
                print('kdt query: elapsed: ', elapsed)

            startIdx = 0
            if N == self.sample_sizes[0]:

                ret += [
                    current_location, current_feature, current_label,
                    current_neighborhood.astype(np.uint32),
                    np.arange(len(current_location), dtype=np.uint32)
                ]
                startIdx = 1

            # TODO: itertools.izip (python2)
            for i, [sample_size, current_k] in enumerate(
                    zip(self.sample_sizes,
                        self.neighborhood_sizes)[startIdx:]):

                #                     print i, sample_size, current_k

                if self.time_dataflow:
                    start_time = timeit.default_timer()

                density = np.sum(current_dist, 1)
                density /= density.sum()

                # cum_density = np.cumsum(density)
                # cum_density /= cum_density[-1]

                # print('cum_density.max(): ', cum_density.max())

                # print 'cum_density: ', cum_density

                subN = sample_size

                # bins = np.linspace(0, 1, self.sample_resolution)
                # inverse = np.digitize(bins, cum_density, right=True)

                idxs = []
                while (len(idxs) < subN):
                    # samples = self.rng.random_sample(2 * subN)
                    # idx = [[
                    #     np.argwhere(cum_density == min(cum_density[(
                    #         cum_density - s) > 0]))
                    # ] for s in samples]
                    # idx = np.unique(idx).tolist()
                    # idxs = np.unique(np.concatenate([idxs, idx]))
                    #             print len(idxs)

                    if self.time_dataflow:
                        start_time = timeit.default_timer()

                    # bins = np.linspace(0, 1, self.sample_resolution)
                    # inverse = np.digitize(bins, cum_density, right=True)
                    # print 'inverse: ', inverse
                    # samples = self.rng.randint(0, self.sample_resolution,
                    #                            2 * subN)

                    idx = self.rng.choice(np.arange(len(density)), 2 * subN, p=density)

                    # print 'samples: ', samples

                    # idx = inverse[samples]

                    idxs = np.concatenate([idxs, idx])

                    unique_indexes = np.unique(idxs, return_index=True)[1]
                    idxs = [idxs[i] for i in sorted(unique_indexes)]

                    if self.time_dataflow:
                        elapsed = timeit.default_timer() - start_time
                        print('sampling: elapsed: ', elapsed)
                        print('len(idxs): ', len(idxs), ' N: ', subN)

                idxs = np.asarray(idxs[:subN], dtype=np.uint32)

                current_location = current_location[idxs]
                current_feature = current_feature[idxs]
                current_label = current_label[idxs]

                kdt = KDTree(
                    current_location, leaf_size=16, metric='euclidean')

                if self.global_density:
                    current_neighborhood = kdt.query(
                        current_location,
                        k=current_k,
                        dualtree=False,
                        return_distance=False)

                    current_dist = current_dist[idxs]
                else:
                    current_dist, current_neighborhood = kdt.query(
                        current_location,
                        k=current_k,
                        dualtree=False,
                        return_distance=True)

                # locations.append(current_location)
                # neighborhoods.append(current_neighborhood)
                # features.append(current_feature)
                # labels.append(current_label)
                # idxs_list.append(idxs)

                ret += [
                    current_location, current_feature, current_label,
                    current_neighborhood.astype(np.uint32), idxs
                ]

            yield ret

    def reset_state(self):
        self.ds.reset_state()
        self.rng = get_rng(self)

    def attributes(self):
        return self.ds.attributes() + ['neighborhood_idx', 'subsample_idxs']

    def size(self):
        return self.ds.size()



