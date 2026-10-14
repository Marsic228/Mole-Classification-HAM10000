class ImageRecord:
    def __init__(self, image_id, lesion_id, dx, dx_type, age, sex, localization):
        self.image_id = image_id
        self.lesion_id = lesion_id
        self.dx = dx
        self.dx_type = dx_type
        self.age = age
        self.sex = sex
        self.localization = localization
        try:
            if isinstance(self.age, str):
                self.age = float(self.age)
        except ValueError:
            self.age = None
        self.dx = self.dx.lower()


    def to_dict(self):
       return {
           "image_id": self.image_id,
           "lesion_id": self.lesion_id,
           "dx": self.dx,
           "dx_type": self.dx_type,
           "age": self.age,
           "sex": self.sex,
           "localization": self.localization
       }
    
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            image_id=data.get("image_id",""),
            lesion_id=data.get("lesion_id", ""),
            dx=data.get("dx", ""),
            dx_type=data.get("dx_type", ""),
            age=data.get("age", ""),
            sex=data.get("sex", ""),
            localization=data.get("localization", "")
        )
    
    def __repr__(self):
        return f"ImageRecord({self.image_id}, dx = '{self.dx}', age = {self.age})"
    
if __name__ == "__main__":
        sample = {
            "image_id": "ISIC_0024306",
            "lesion_id": "HAM_0000001",
            "dx": "MEL",
            "dx_type": "histo",
            "age": "55.0",
            "sex": "male",
            "localization": "lower extremity"
            }

        record = ImageRecord.from_dict(sample)
        print(record)
        print(record.to_dict())
    
