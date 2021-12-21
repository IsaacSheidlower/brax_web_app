
import os

class uuid_manager(object):
    def __init__(self, filename="uuids.txt"):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.fn = self.path+"/"+filename
        self.uuid_idx = 0
        self.uuid_count_fn = self.path+"/uuid_count.txt"
        with open(self.uuid_count_fn) as fp:
            self.uuid_idx = int(fp.readline())

        print("UUID manager will read {} from index {}.".format(self.fn, self.uuid_idx))

    def get_uuid(self):
        ret = "invalid"
        # get the uuid without reading the file into memory
        with open(self.fn) as fp:
            for i, line in enumerate(fp):
                if i == self.uuid_idx:
                    ret = line.split("\n")[0]
                    print("Reading {} from line {}".format(ret, i))
                    break

        if ret != "invalid":
            self.uuid_idx += 1
            with open(self.uuid_count_fn, "w") as fp:
                fp.write(str(self.uuid_idx))

        return ret

if __name__ == "__main__":
    mgr = uuid_manager()
    for i in range(100):
        print(mgr.get_uuid())
