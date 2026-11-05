"""Phase 2 contracts and controlled reference checks; never imports legacy runners."""
import ast
from contextlib import chdir, redirect_stdout, redirect_stderr, contextmanager
import csv
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def legacy_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@contextmanager
def fixture_directory():
    # Retain isolated fixtures; no automatic verification-artifact cleanup.
    yield tempfile.mkdtemp(prefix="phase2-fixture-")


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module("evaluation.compare_predicted_roi")

    def test_arguments_and_fold_paths(self):
        for bad in ([], ["--fold", "0"], ["--fold", "6"], ["--fold", "all"]):
            with self.subTest(bad=bad), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.m.parser().parse_args(bad)
        for fold in range(1, 6):
            args = self.m.parser().parse_args(["--fold", str(fold)])
            paths = self.m.resolve_paths(args)
            self.assertEqual(paths["checkpoint"], ROOT / f"checkpoints/{self.m.RUN}_fold{fold}_best.pt")
            self.assertEqual(paths["predicted_root"], ROOT / f"data/processed_roi_predicted_fold{fold}_v1")
            self.assertEqual(paths["output"].name, f"classifier_predicted_vs_reference_fold{fold}.csv")
        with chdir(ROOT.parent):
            self.assertEqual(self.m.resolve_paths(args)["reference_root"], ROOT / "data/processed_roi_margin15_v1")

    def test_collision_before_metadata_or_model(self):
        with fixture_directory() as directory:
            args = self.m.parser().parse_args(["--fold", "1", "--output-root", directory])
            paths = self.m.resolve_paths(args)
            paths["output"].touch()
            with patch.object(self.m, "load_metadata", side_effect=AssertionError("metadata reached")):
                with self.assertRaises(FileExistsError):
                    self.m.preflight(args, paths)

    def test_dry_run_has_no_model_or_checkpoint_load(self):
        import torch
        with fixture_directory() as directory:
            output = Path(directory) / "absent"
            with patch.object(self.m, "build_model", side_effect=AssertionError("model")), patch.object(torch, "load", side_effect=AssertionError("load")), redirect_stdout(io.StringIO()):
                self.m.main(["--fold", "1", "--output-root", str(output), "--dry-run", "--device", "cpu"])
            self.assertFalse(output.exists())

    def test_canonical_ids_order_classes_and_labels(self):
        from core.canonical_cv import get_fold_metadata, verified_assignments
        from core.cross_validation_dataset import CrossValidationDataset
        with chdir(ROOT):
            assignments, counts = verified_assignments()
            for fold in range(1, 6):
                train, original = get_fold_metadata(fold)
                val = self.m.load_metadata(fold)
                self.assertEqual(list(val.image_id), list(original.image_id))
                self.assertEqual(set(val.image_id), {k for k, r in assignments.items() if r["fold"] == fold})
                self.assertEqual(len(val), counts[fold]["val"])
                self.assertFalse(set(train.lesion_id) & set(val.lesion_id))
                saved = list(csv.DictReader(io.StringIO((ROOT / f"reports/segmentation/small_unet_fold{fold}_seed123/classifier_predicted_vs_reference_fold{fold}.csv").read_text(encoding="utf-8"))))
                self.assertEqual(list(val.image_id), [r["image_id"] for r in saved])
                ref = CrossValidationDataset(val, ROOT / "data/processed_roi_margin15_v1")
                pred_meta = val.copy(); pred_meta["source_split"] = "val"
                pred = CrossValidationDataset(pred_meta, ROOT / f"data/processed_roi_predicted_fold{fold}_v1")
                self.m.validate_pairs(val, ref, pred)
                self.assertEqual(ref.targets, [int(r["true_label"]) for r in saved])
                pred.samples[0] = (pred.samples[0][0], (pred.samples[0][1] + 1) % 7)
                with self.assertRaises(ValueError):
                    self.m.validate_pairs(val, ref, pred)
                pred.classes = list(reversed(CLASSES))
                with self.assertRaises(ValueError):
                    self.m.validate_pairs(val, ref, pred)

    def test_wrong_checkpoint_missing_inputs_and_report(self):
        with fixture_directory() as directory:
            args = self.m.parser().parse_args(["--fold", "1", "--output-root", directory, "--checkpoint", str(Path(directory)/"missing")])
            with self.assertRaises(FileNotFoundError):
                self.m.preflight(args, self.m.resolve_paths(args))
            wrong = Path(directory)/"wrong.pt"; wrong.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                self.m.verify_checkpoint(wrong, 1)
            args.checkpoint = ROOT / f"checkpoints/{self.m.RUN}_fold2_best.pt"
            with self.assertRaises(ValueError):
                self.m.preflight(args, self.m.resolve_paths(args))

    def test_legacy_evaluate_incomplete_batch(self):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(991)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 7)).eval()
        device = torch.device("cpu")
        dataset = TensorDataset(torch.randn(11, 3, 2, 2), torch.arange(11) % 7)
        original = legacy_function("evaluation/compare_predicted_roi_fold1.py", "evaluate", dict(torch=torch, np=np, DataLoader=DataLoader, model=model, device=device))
        old_y, old_p = original(dataset)
        y, p, probabilities = self.m.evaluate(dataset, model, device, capture_probabilities=True)
        np.testing.assert_array_equal(y, old_y)
        np.testing.assert_array_equal(p, old_p)
        self.assertEqual(probabilities.shape, (11, 7))

    def test_defaults_transform_and_source_families(self):
        from torchvision.models import ConvNeXt_Tiny_Weights
        t = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        self.assertEqual(t.crop_size, [224]); self.assertEqual(t.resize_size, [236])
        self.assertEqual(t.mean, [0.485, 0.456, 0.406])
        self.assertEqual(t.std, [0.229, 0.224, 0.225])
        # Strip only known fold/count/comment/whitespace differences.
        import re
        def normalize(text):
            text = re.sub(r"fold[1-5]|Fold [1-5]", "foldN", text)
            text = re.sub(r"get_fold_metadata\([1-5]\)", "get_fold_metadata(N)", text)
            text = re.sub(r"== (1707|1706|len\(val\))", "== COUNT", text)
            return [" ".join(line.split()) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        base = normalize((ROOT / "evaluation/compare_predicted_roi_fold1.py").read_text())
        for fold in range(2, 6):
            self.assertEqual(base, normalize((ROOT / f"evaluation/compare_predicted_roi_fold{fold}.py").read_text()))


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module("evaluation.export_predicted_roi")

    def test_arguments_paths_and_empty_root_collision(self):
        for fold in range(1, 6):
            args = self.m.parser().parse_args(["--fold", str(fold)])
            paths = self.m.resolve_paths(args)
            self.assertEqual(paths["checkpoint"], ROOT / f"checkpoints/segmentation/small_unet_fold{fold}_seed123_best.pt")
            self.assertEqual(paths["output_root"], ROOT / f"data/processed_roi_predicted_fold{fold}_v1")
        with fixture_directory() as directory:
            args.output_root = Path(directory)
            with patch("torch.load", side_effect=AssertionError("load")), self.assertRaises(FileExistsError):
                self.m.preflight(args, self.m.resolve_paths(args))
        for argv in ([], ["--fold", "0"], ["--fold", "all"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.m.parser().parse_args(argv)

    def test_dry_run_and_missing_input(self):
        with fixture_directory() as directory:
            target = Path(directory)/"absent"
            with patch.object(self.m, "build_model", side_effect=AssertionError("model")), patch("torch.load", side_effect=AssertionError("load")), redirect_stdout(io.StringIO()):
                self.m.main(["--fold", "2", "--output-root", str(target), "--dry-run", "--device", "cpu"])
            self.assertFalse(target.exists())
            args = self.m.parser().parse_args(["--fold", "2", "--output-root", str(target), "--image-root", str(target)])
            with self.assertRaises(FileNotFoundError):
                self.m.preflight(args, self.m.resolve_paths(args))

    def test_crop_fallbacks_and_geometry(self):
        import numpy as np
        from PIL import Image
        from preprocessing.roi_crop import crop_lesion_with_margin
        image = Image.fromarray(np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3))
        cases = []
        empty=np.zeros((80,100),np.uint8); cases.append((empty,"fallback_empty"))
        large=np.full((80,100),255,np.uint8); cases.append((large,"fallback_large_area"))
        edge=empty.copy(); edge[0,50]=255; cases.append((edge,"fallback_edge"))
        wide=empty.copy(); wide[20:60,10:90]=255; cases.append((wide,"fallback_large_crop"))
        roi=empty.copy(); roi[35:45,45:55]=255; cases.append((roi,"roi"))
        for array,status in cases:
            cropped, actual = crop_lesion_with_margin(image,Image.fromarray(array))
            self.assertEqual(actual,status)
            if status.startswith("fallback"):
                np.testing.assert_array_equal(np.asarray(cropped),np.asarray(image))
            else:
                self.assertEqual(cropped.size,(35,28))
        boundary=empty.copy(); boundary[:48,:]=255
        self.assertEqual(crop_lesion_with_margin(image,Image.fromarray(boundary))[1],"fallback_edge")
        boundary[48,0]=255
        self.assertEqual(crop_lesion_with_margin(image,Image.fromarray(boundary))[1],"fallback_large_area")
        with self.assertRaises(ValueError):
            crop_lesion_with_margin(image,Image.new("L",(1,1)))

    def test_original_tensor_mask_and_crop_expressions(self):
        import numpy as np
        import torch
        from PIL import Image
        from preprocessing.roi_crop import crop_lesion_with_margin
        source=ast.parse((ROOT/"evaluation/export_predicted_roi_fold1.py").read_text())
        block=next(n for n in source.body if isinstance(n,ast.With)).body[0].body
        start=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="resized")
        end=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))
        ns=dict(np=np,torch=torch,Image=Image,crop_lesion_with_margin=crop_lesion_with_margin)
        device=torch.device("cpu")
        class FixtureModel(torch.nn.Module):
            def forward(self,x):
                return x[:,:1]-0.5
        model=FixtureModel(); image=Image.fromarray(np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3))
        ns.update(device=device,model=model,image=image)
        with torch.inference_mode():
            exec(compile(ast.Module(body=block[start:end+1],type_ignores=[]),"original_export_expressions","exec"),ns)
            cropped,status,predicted,mask=self.m.predict_crop(image,model,device)
        np.testing.assert_array_equal(predicted,ns["predicted"])
        np.testing.assert_array_equal(np.asarray(mask),np.asarray(ns["mask"]))
        np.testing.assert_array_equal(np.asarray(cropped),np.asarray(ns["cropped"]))
        self.assertEqual(status,ns["status"])
        class ZeroModel(torch.nn.Module):
            def forward(self,x): return torch.zeros((1,1,192,256))
        with torch.inference_mode():
            self.assertTrue(self.m.predict_crop(image,ZeroModel(),device)[2].all())

    def test_exact_output_validation(self):
        import pandas as pd
        from PIL import Image
        val=pd.DataFrame([dict(image_id="a",dx="mel"),dict(image_id="b",dx="nv")])
        with fixture_directory() as directory:
            output=Path(directory)
            for row in val.itertuples():
                path=output/"val"/row.dx/f"{row.image_id}_224x224.jpg"
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(224,224)).save(path)
            self.m.validate_outputs(output,val)
            (output/"extra.jpg").write_bytes(b"unexpected")
            with self.assertRaises(ValueError): self.m.validate_outputs(output,val)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module("experiments.segmentation.train")

    def test_arguments_paths_and_collision(self):
        for fold in range(1,6):
            args=self.m.parser().parse_args(["--fold",str(fold)])
            paths=self.m.resolve_paths(args)
            self.assertEqual(paths["checkpoint"],ROOT/f"checkpoints/segmentation/small_unet_fold{fold}_seed123_best.pt")
            self.assertEqual(paths["report_dir"],ROOT/f"reports/segmentation/small_unet_fold{fold}_seed123")
            with self.assertRaises(FileExistsError): self.m.preflight(args,paths)
        for argv in ([],["--fold","0"],["--fold","all"]):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): self.m.parser().parse_args(argv)

    def test_dry_run_and_missing_inputs(self):
        with fixture_directory() as directory:
            args=self.m.parser().parse_args(["--fold","1","--output-root",str(Path(directory)/"new")])
            with patch.object(self.m,"initialize",side_effect=AssertionError("initialize")),patch("torch.load",side_effect=AssertionError("load")),redirect_stdout(io.StringIO()):
                self.m.main(["--fold","1","--output-root",str(Path(directory)/"new"),"--dry-run","--device","cpu"])
            self.assertFalse((Path(directory)/"new").exists())
            args.mask_mapping=Path(directory)/"missing"
            with self.assertRaises(FileNotFoundError): self.m.preflight(args,self.m.resolve_paths(args))

    def test_initialization_rng_and_loader_order(self):
        import torch
        source=ast.parse((ROOT/"experiments/segmentation/train_fold1.py").read_text())
        prefix=source.body[:next(i for i,n in enumerate(source.body) if isinstance(n,ast.FunctionDef))]
        ns={}
        with chdir(ROOT),patch.object(torch.cuda,"is_available",return_value=False):
            exec(compile(ast.Module(body=prefix,type_ignores=[]),"original_training_initialization","exec"),ns)
        original_rng=torch.get_rng_state().clone()
        original_params={k:v.clone() for k,v in ns["model"].state_dict().items()}
        original_opt=ns["optimizer"].state_dict()
        iterator=iter(ns["train_loader"])
        original_ids=[next(iterator)[2] for _ in range(2)]
        original_after=torch.get_rng_state().clone()
        args=self.m.parser().parse_args(["--fold","1","--device","cpu"])
        state=self.m.initialize(args,self.m.resolve_paths(args))
        model,optimizer,train_loader,val_loader,val_dataset,device=state
        self.assertTrue(torch.equal(original_rng,torch.get_rng_state()))
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,original_params[k]),k)
        self.assertEqual(optimizer.state_dict(),original_opt)
        iterator=iter(train_loader)
        self.assertEqual([next(iterator)[2] for _ in range(2)],original_ids)
        self.assertTrue(torch.equal(original_after,torch.get_rng_state()))
        self.assertEqual(train_loader.batch_size,4); self.assertEqual(train_loader.num_workers,0)
        self.assertEqual(val_loader.batch_size,4); self.assertEqual(val_loader.num_workers,0)
        self.assertFalse(train_loader.drop_last); self.assertFalse(val_loader.drop_last)
        self.assertEqual(list(val_dataset.metadata.image_id),list(ns["val_dataset"].metadata.image_id))

    def test_two_synthetic_steps_match_original(self):
        import copy
        import torch
        from torch.utils.data import DataLoader
        from core.segmentation_losses import segmentation_loss
        from core.segmentation_unet import SmallUNet
        torch.set_num_threads(1)
        torch.manual_seed(321)
        original=SmallUNet(); consolidated=copy.deepcopy(original)
        old_opt=torch.optim.Adam(original.parameters(),lr=1e-3)
        new_opt=torch.optim.Adam(consolidated.parameters(),lr=1e-3)
        images=torch.rand(5,3,16,16); masks=(torch.rand(5,1,16,16)>=0.5).float()
        dataset=[(images[i],masks[i],f"synthetic_{i}") for i in range(5)]
        old_loop=legacy_function("experiments/segmentation/train_fold1.py","train_one_epoch",dict(torch=torch,segmentation_loss=segmentation_loss))
        with redirect_stdout(io.StringIO()):
            old_loss=old_loop(original,DataLoader(dataset,batch_size=4),old_opt,torch.device("cpu"))
            new_loss=self.m.train_one_epoch(consolidated,DataLoader(dataset,batch_size=4),new_opt,torch.device("cpu"))
        self.assertEqual(old_loss,new_loss)
        for k,v in original.state_dict().items(): self.assertTrue(torch.equal(v,consolidated.state_dict()[k]),k)
        for k,values in old_opt.state_dict()["state"].items():
            for name,value in values.items(): self.assertTrue(torch.equal(value,new_opt.state_dict()["state"][k][name]))
        # Compare the complete original function AST except lazy imports.
        old_node=ast.parse((ROOT/"experiments/segmentation/train_fold1.py").read_text())
        old_node=next(n for n in old_node.body if isinstance(n,ast.FunctionDef))
        new_node=ast.parse((ROOT/"experiments/segmentation/train.py").read_text())
        new_node=next(n for n in new_node.body if isinstance(n,ast.FunctionDef) and n.name=="train_one_epoch")
        new_node.body=[n for n in new_node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertEqual(ast.dump(old_node,include_attributes=False),ast.dump(new_node,include_attributes=False))

    def test_selection_patience_epoch_cap_and_checkpoint_schema(self):
        import torch
        from types import SimpleNamespace
        model=SimpleNamespace(state_dict=lambda:{"weight":torch.tensor([1.])})
        optimizer=SimpleNamespace(state_dict=lambda:{"state":{},"param_groups":[]})
        for scores,expected_epoch,expected_length in (([0.5,0.5001,0.5001,0.5001,0.5001],1,5),([0.5+i*0.001 for i in range(12)],12,12),([0.5,0.50010001]+[0.50010001]*4,2,6)):
            reports=[({"n_images":1,"mean_dice":s,"mean_iou":s,"empty_predictions":0},[{"image_id":"synthetic","dice":s,"iou":s,"predicted_empty":False}]) for s in scores]
            with fixture_directory() as directory:
                report_dir=Path(directory)/"reports"; report_dir.mkdir()
                captured=[]
                with patch.object(self.m,"train_one_epoch",return_value=0.2),patch("evaluation.segmentation_validation.evaluate_segmentation",side_effect=reports),patch.object(torch,"save",side_effect=lambda payload,path:captured.append(payload)),redirect_stdout(io.StringIO()):
                    epoch,dice=self.m.fit(model,optimizer,None,None,[0],torch.device("cpu"),report_dir,Path(directory)/"best.pt",3)
                self.assertEqual(epoch,expected_epoch)
                self.assertEqual(len(json.loads((report_dir/"history.json").read_text())),expected_length)
                self.assertEqual(captured[-1]["epoch"],expected_epoch)
                self.assertEqual(set(captured[-1]),{"model_state_dict","optimizer_state_dict","epoch","fold","seed","mean_dice","config"})
                self.assertEqual(captured[-1]["fold"],3); self.assertEqual(captured[-1]["seed"],123)
                self.assertEqual(captured[-1]["config"],dict(input_size=[192,256],batch_size=4,learning_rate=1e-3,loss="BCE + soft Dice",threshold=0.5))


class EntryPointTests(unittest.TestCase):
    def test_syntax_and_import_without_computation(self):
        import torch
        import subprocess
        import sys
        modules=("evaluation.compare_predicted_roi","evaluation.export_predicted_roi","experiments.segmentation.train")
        for name in modules:
            module=importlib.import_module(name)
            ast.parse(Path(module.__file__).read_text())
            with patch.object(torch,"load",side_effect=AssertionError("load")),patch.object(module,"run",side_effect=AssertionError("run")):
                importlib.reload(module)
            result=subprocess.run([sys.executable,"-B","-m",name,"--help"],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("--fold",result.stdout)


class DeviceAndContractTests(unittest.TestCase):
    def test_device_errors_and_exact_cli_surface(self):
        import torch
        expected={
            "evaluation.compare_predicted_roi":{"fold","checkpoint","reference_root","predicted_root","reference_report","output_root","device","dry_run","help"},
            "evaluation.export_predicted_roi":{"fold","checkpoint","image_root","output_root","device","dry_run","help"},
            "experiments.segmentation.train":{"fold","image_root","mask_mapping","mask_archive","output_root","device","dry_run","help"},
        }
        for name,arguments in expected.items():
            module=importlib.import_module(name)
            self.assertEqual({a.dest for a in module.parser()._actions},arguments)
            with patch.object(torch.cuda,"is_available",return_value=False):
                self.assertEqual(module.select_device("auto").type,"cpu")
                with self.assertRaises(ValueError): module.select_device("cuda")

    def test_original_training_and_export_family_differences(self):
        import re
        def normalized(text):
            text=re.sub(r"fold[1-5]|Fold [1-5]","foldN",text)
            text=re.sub(r"get_fold_metadata\([1-5]\)","get_fold_metadata(N)",text)
            text=re.sub(r'"fold": [1-5]','"fold": N',text)
            text=re.sub(r"== (1707|1706|len\(val\))","== COUNT",text)
            return [" ".join(l.split()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        for prefix in ("experiments/segmentation/train","evaluation/export_predicted_roi"):
            original=normalized((ROOT/f"{prefix}_fold1.py").read_text())
            for fold in range(2,6):self.assertEqual(original,normalized((ROOT/f"{prefix}_fold{fold}.py").read_text()))


if __name__ == "__main__":
    unittest.main()
