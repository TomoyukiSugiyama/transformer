# Phase 2: Tiny Shakespeare 推論サンプル (best checkpoint: step 2500)

Accelerate 版 (`phase2_d256_ff1024_max128_with_accelerate`) の **step 2500** チェックポイントを
`top_k=5, temperature=1.0, repetition_penalty=1.2, max_new_token=100` で生成した結果。
学習開始から約 29 分で到達した、 loss=2.785 / perplexity≈16 の状態。

各プロンプトが **異なる作品にきれいに着地**しているのが特徴 (Measure for Measure / Romeo & Juliet /
Winter's Tale / Coriolanus と 4 作品が分散)。 単に登場人物名が合っているだけでなく、 **シーンのトーン
(神学・恋愛・田舎喜劇・政治) もそれぞれの作品らしさ**を持っている。

## prompt: "I have seen" → Measure for Measure

```
i have seen.

duke vincentio:
your grace of justice, sir; your honour is worth the friar,
and you shall be satisfied withal.

isabella:
but, as if you were a father had so long
that in the fault have found it, which should not
to make his instantify him that he would
hear me give my life.

duke vincentio:
this is angelo for the world with him
that he hath made good success and cut away
```

## prompt: "O Romeo" → Romeo & Juliet

```
o romeo, thou art not proud;
thy lips might have been so many a man:
the good captain of death is in his way,
to help me and romeo from whence are come,
and i will be long with my tent there.
i am too far in my heart, to make thee think
where he did send to him to me at me;
and let him speak no better it as myself.

juliet:
but, by your grace of joy!
```

## prompt: "To be or not to be" → Winter's Tale

```
to be or not to be long.

autolycus:
i shall be of it, sir: and you know the
way, sir, have your voices.

clown:
i do beseech your worship, give no leave to be so.

shepherd:
are you now, good morrow; and, indeed, for your daughter?

autolycus:
i will be a thousand of it, if it be but
done, it is a mach.

clown:
how could you
```

## prompt: "What news" → Coriolanus

```
what news abroad, i
to give my vow; but come on your highness' hands
i would not have heard the crown.

second gentleman:
i know no longer, i am sorry that he had so
discreep o'er again: therefore i'll do you
more than you in this.

first senator:
i will not hence.

brutus:
go along with us.

menenius:
nay, good madam, go.
i have a soldier
```

## この checkpoint からの推論方法

> **重要**: `checkpoints/` は `.gitignore` 対象なので、 リポジトリには含まれていません。
> 推論するには **先に同 Config で `training_and_inference` を完走させて** checkpoint を
> 自前で生成する必要があります。 詳細は [README の checkpoint 運用](../README.md#checkpoint-から再開推論) を参照。

```rust
// src/main.rs
fn main() {
    let cfg = Config::tiny_shakespeare();
    // 1) まず学習を完走させて checkpoints/<run_name>/best.bin を生成:
    // training_and_inference(&cfg);

    // 2) 完走後にこちらに切替えて推論のみを実行:
    inference_from_checkpoint(
        &cfg,
        "checkpoints/phase2_d256_ff1024_max128_with_accelerate/step_002500.bin",
    );
}
```
