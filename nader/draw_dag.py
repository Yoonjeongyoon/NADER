import os
from ModelFactory.block_factory import BlockFactory
# block_factory.py (BlockFactory 클래스 안에 추가)
# draw_dag.py
import os
import glob
import argparse
from ModelFactory.block_factory import BlockFactory

def visualize_txt(bf, txt, *, out_dir=".", file_prefix=None, draw_shapes=True):
    """
    BlockFactory 수정을 하지 않고, 기존 메서드(parse_txt, get_shape, drawDAG)만 사용해서
    PNG 2종(기본/shape라벨)을 생성하는 가벼운 헬퍼.

    Parameters
    ----------
    bf : BlockFactory
        기존 block_factory의 인스턴스
    txt : str
        NADER DAG 텍스트 정의
    out_dir : str
        이미지 저장 폴더
    file_prefix : str or None
        출력 파일 접두사 (None이면 dag['name'])
    draw_shapes : bool
        True면 node2output 계산해서 shape 라벨 버전도 시도

    Returns
    -------
    dict
        {"basic_png": str, "shape_png": str|None, "used_shapes": bool}
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) txt -> dag(dict)
    dag = bf.parse_txt(txt)

    # 2) 파일명 결정
    base = file_prefix or dag.get('name', 'dag_preview')
    basic_name = base
    shape_name = f"{base}_shape"

    # 3) 기본 그래프
    bf.drawDAG(dag, file_name=basic_name, file_dir=out_dir)
    basic_png = os.path.join(out_dir, f"{basic_name}.png")

    # 4) shape 라벨 그래프 (가능하면)
    used_shapes, shape_png = False, None
    if draw_shapes:
        try:
            node2shape, node2output = bf.get_shape(dag)  # (node2shape, node2output) 가정
            dag['node2shape'] = node2shape
            dag['node2output'] = node2output
            # drawDAG가 Params/FLOPs 라벨을 요구할 수 있으니 더미값 채움
            dag.setdefault('params', '-')
            dag.setdefault('flops',  '-')

            bf.drawDAG(dag, file_name=shape_name, file_dir=out_dir, draw_edge_label=True)
            shape_png = os.path.join(out_dir, f"{shape_name}.png")
            used_shapes = True
        except Exception:
            # shape 계산 실패해도 기본 이미지는 이미 생성됨
            shape_png = None
            used_shapes = False

    return {"basic_png": basic_png, "shape_png": shape_png, "used_shapes": used_shapes}


def render_from_dir(txt_dir, out_dir, draw_shapes=True, pattern="*.txt"):
    """
    txt_dir 안의 모든 txt 파일을 순회하며 이미지 생성
    """
    os.makedirs(out_dir, exist_ok=True)
    bf = BlockFactory()  # block_factory는 수정하지 않음

    paths = sorted(glob.glob(os.path.join(txt_dir, pattern)))
    if not paths:
        print(f"[WARN] No txt files matched in: {txt_dir}/{pattern}")
        return

    for p in paths:
        with open(p, "r") as f:
            txt = f.read()
        prefix = os.path.splitext(os.path.basename(p))[0]
        print(f"[INFO] Rendering {prefix} ...")
        res = visualize_txt(bf, txt, out_dir=out_dir, file_prefix=prefix, draw_shapes=draw_shapes)
        print(f"  ✅ Basic : {res['basic_png']}")
        if res["shape_png"]:
            print(f"  ✅ Shape : {res['shape_png']}")
        else:
            print("  ⚠️  Shape version skipped.")


def main():
    parser = argparse.ArgumentParser(description="Render DAG PNGs from NADER txt files (no BlockFactory edits).")
    parser.add_argument("--txt", type=str, default=None, help="단일 txt 파일 경로")
    parser.add_argument("--txt_dir", type=str, default=None, help="txt 파일들이 모인 디렉터리")
    parser.add_argument("--out_dir", type=str, default="./fpn_images", help="이미지 출력 폴더")
    parser.add_argument("--no-shapes", action="store_true", help="shape 라벨 이미지를 만들지 않음")
    parser.add_argument("--pattern", type=str, default="*.txt", help="txt_dir에서 매칭할 패턴 (기본: *.txt)")
    args = parser.parse_args()

    draw_shapes = not args.no_shapes

    if args.txt:
        # 단일 파일
        with open(args.txt, "r") as f:
            txt = f.read()
        bf = BlockFactory()
        prefix = os.path.splitext(os.path.basename(args.txt))[0]
        res = visualize_txt(bf, txt, out_dir=args.out_dir, file_prefix=prefix, draw_shapes=draw_shapes)
        print(f"✅ Basic : {res['basic_png']}")
        if res["shape_png"]:
            print(f"✅ Shape : {res['shape_png']}")
        else:
            print("⚠️ Shape version skipped.")
    elif args.txt_dir:
        # 디렉터리 일괄 처리
        render_from_dir(args.txt_dir, args.out_dir, draw_shapes=draw_shapes, pattern=args.pattern)
    else:
        print("사용법: --txt <파일> 또는 --txt_dir <폴더> 중 하나를 지정하세요.")

if __name__ == "__main__":
    main()
