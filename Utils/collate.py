def collate(batch):
    imgs, targs = zip(*batch)
    return list(imgs), list(targs)


